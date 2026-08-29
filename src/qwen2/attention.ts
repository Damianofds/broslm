import type { Qwen2LayerKvCache } from "./attentionCache";
import type { Qwen2AttentionWeights, QwenTensorView, TensorView } from "./loader";
import { matrixVectorMultiplyQwen, matrixVectorMultiplyQwenGpu } from "./quantizedTensor";
import { applyRoPE, applyRoPEGpu } from "../primitives/applyRoPE";
import { softmax } from "../primitives/softmax";
import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export interface Qwen2SelfAttentionConfig {
  hiddenSize: number;
  numberOfHeads: number;
  numberOfKeyValueHeads: number;
  headDimension: number;
  keyValueHiddenSize: number;
  ropeTheta: number;
}

export interface Qwen2SelfAttentionWeights {
  query: QwenTensorView;
  queryBias?: TensorView | Float32Array;
  key: QwenTensorView;
  keyBias?: TensorView | Float32Array;
  value: QwenTensorView;
  valueBias?: TensorView | Float32Array;
  output: QwenTensorView;
}

export interface Qwen2SelfAttentionDebug {
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  rawScores: Float32Array;
  scaledScores: Float32Array;
  probabilities: Float32Array;
  headOutput: Float32Array;
  concatenated: Float32Array;
  projected: Float32Array;
}

export interface Qwen2SelfAttentionResult {
  output: Float32Array;
  debug: Qwen2SelfAttentionDebug;
}

export function qwen2SelfAttention(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
): Float32Array {
  return qwen2SelfAttentionWithDebug(input, sequenceLength, config, weights).output;
}

export async function qwen2SelfAttentionGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
): Promise<Float32Array> {
  return (await qwen2SelfAttentionWithDebugGpu(runtime, input, sequenceLength, config, weights)).output;
}

export function qwen2SelfAttentionPrefill(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
  cache: Qwen2LayerKvCache,
): Float32Array {
  const result = qwen2SelfAttentionWithDebug(input, sequenceLength, config, weights);
  writeProjectedKvToCache(result.debug.k, result.debug.v, sequenceLength, config, cache);
  return result.output;
}

export async function qwen2SelfAttentionPrefillGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
  cache: Qwen2LayerKvCache,
): Promise<Float32Array> {
  const result = await qwen2SelfAttentionWithDebugGpu(runtime, input, sequenceLength, config, weights);
  writeProjectedKvToCache(result.debug.k, result.debug.v, sequenceLength, config, cache);
  return result.output;
}

export function qwen2SelfAttentionIncremental(
  input: Float32Array,
  position: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
  cache: Qwen2LayerKvCache,
): Float32Array {
  validateAttentionConfig(config);
  validateAttentionWeights(config, weights);
  if (input.length !== config.hiddenSize) {
    throw new Error(`input length is ${input.length}, expected ${config.hiddenSize}`);
  }
  if (cache.length !== position) {
    throw new Error(`cache length is ${cache.length}, expected current position ${position}`);
  }
  validateCacheBuffers(cache, config);

  const { hiddenSize, keyValueHiddenSize, numberOfHeads, headDimension } = config;
  const q = new Float32Array(hiddenSize);
  const k = new Float32Array(keyValueHiddenSize);
  const v = new Float32Array(keyValueHiddenSize);
  const concatenated = new Float32Array(hiddenSize);
  const projected = new Float32Array(hiddenSize);
  const scores = new Float32Array(position + 1);
  const probabilities = new Float32Array(position + 1);

  matrixVectorMultiplyQwen(weights.query, input, q, { bias: weights.queryBias });
  matrixVectorMultiplyQwen(weights.key, input, k, { bias: weights.keyBias });
  matrixVectorMultiplyQwen(weights.value, input, v, { bias: weights.valueBias });
  applyRoPEToQuery(q, 1, config, position);
  applyRoPEToKey(k, 1, config, position);

  const cacheOffset = position * keyValueHiddenSize;
  cache.keys.set(k, cacheOffset);
  cache.values.set(v, cacheOffset);

  const scale = 1 / Math.sqrt(headDimension);
  for (let head = 0; head < numberOfHeads; head += 1) {
    const keyValueHead = keyValueHeadForQueryHead(head, config);
    for (let keyPosition = 0; keyPosition <= position; keyPosition += 1) {
      let score = 0;
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        score +=
          (q[head * headDimension + dimension] ?? 0) *
          (cache.keys[kvHiddenIndex(keyPosition, keyValueHead, dimension, config)] ?? 0);
      }
      scores[keyPosition] = score * scale;
    }

    softmax(scores, probabilities, {
      length: position + 1,
    });

    for (let dimension = 0; dimension < headDimension; dimension += 1) {
      let value = 0;
      for (let valuePosition = 0; valuePosition <= position; valuePosition += 1) {
        value +=
          (probabilities[valuePosition] ?? 0) *
          (cache.values[kvHiddenIndex(valuePosition, keyValueHead, dimension, config)] ?? 0);
      }
      concatenated[head * headDimension + dimension] = value;
    }
  }

  matrixVectorMultiplyQwen(weights.output, concatenated, projected);
  cache.length = position + 1;
  return projected;
}

export async function qwen2SelfAttentionIncrementalGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  position: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
  cache: Qwen2LayerKvCache,
): Promise<Float32Array> {
  validateAttentionConfig(config);
  validateAttentionWeights(config, weights);
  if (input.length !== config.hiddenSize) {
    throw new Error(`input length is ${input.length}, expected ${config.hiddenSize}`);
  }
  if (cache.length !== position) {
    throw new Error(`cache length is ${cache.length}, expected current position ${position}`);
  }
  validateCacheBuffers(cache, config);

  const q = await matrixVectorMultiplyQwenGpu(runtime, weights.query, input, {
    bias: weights.queryBias,
  });
  const k = await matrixVectorMultiplyQwenGpu(runtime, weights.key, input, {
    bias: weights.keyBias,
  });
  const v = await matrixVectorMultiplyQwenGpu(runtime, weights.value, input, {
    bias: weights.valueBias,
  });
  await applyRoPEToQueryGpu(runtime, q, 1, config, position);
  await applyRoPEToKeyGpu(runtime, k, 1, config, position);

  const cacheOffset = position * config.keyValueHiddenSize;
  cache.keys.set(k, cacheOffset);
  cache.values.set(v, cacheOffset);

  const qSequence = new Float32Array((position + 1) * config.hiddenSize);
  qSequence.set(q, position * config.hiddenSize);
  const concatenated = await causalGqaHeadOutputsGpu(runtime, {
    q: qSequence,
    k: cache.keys.subarray(0, (position + 1) * config.keyValueHiddenSize),
    v: cache.values.subarray(0, (position + 1) * config.keyValueHiddenSize),
    sequenceLength: position + 1,
    config,
    outputLastPositionOnly: true,
  });
  const projected = await matrixVectorMultiplyQwenGpu(runtime, weights.output, concatenated);
  cache.length = position + 1;
  return projected;
}

export function qwen2SelfAttentionWithDebug(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
): Qwen2SelfAttentionResult {
  validateAttentionInputs(input, sequenceLength, config, weights);

  const { hiddenSize, keyValueHiddenSize, numberOfHeads } = config;
  const hiddenElements = sequenceLength * hiddenSize;
  const keyValueElements = sequenceLength * keyValueHiddenSize;
  const scoreElements = numberOfHeads * sequenceLength * sequenceLength;
  const q = new Float32Array(hiddenElements);
  const k = new Float32Array(keyValueElements);
  const v = new Float32Array(keyValueElements);
  const rawScores = new Float32Array(scoreElements);
  const scaledScores = new Float32Array(scoreElements);
  const probabilities = new Float32Array(scoreElements);
  const headOutput = new Float32Array(hiddenElements);
  const concatenated = new Float32Array(hiddenElements);
  const projected = new Float32Array(hiddenElements);

  rawScores.fill(Number.NaN);
  scaledScores.fill(Number.NaN);
  probabilities.fill(Number.NaN);

  projectSequence(input, sequenceLength, hiddenSize, weights.query, q, hiddenSize, weights.queryBias);
  projectSequence(
    input,
    sequenceLength,
    hiddenSize,
    weights.key,
    k,
    keyValueHiddenSize,
    weights.keyBias,
  );
  projectSequence(
    input,
    sequenceLength,
    hiddenSize,
    weights.value,
    v,
    keyValueHiddenSize,
    weights.valueBias,
  );
  applyRoPEToQuery(q, sequenceLength, config);
  applyRoPEToKey(k, sequenceLength, config);

  computeCausalGqaHeadOutputs({
    q,
    k,
    v,
    rawScores,
    scaledScores,
    probabilities,
    headOutput,
    concatenated,
    sequenceLength,
    config,
  });

  projectSequence(concatenated, sequenceLength, hiddenSize, weights.output, projected, hiddenSize);

  return {
    output: projected,
    debug: {
      q,
      k,
      v,
      rawScores,
      scaledScores,
      probabilities,
      headOutput,
      concatenated,
      projected,
    },
  };
}

export async function qwen2SelfAttentionWithDebugGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
): Promise<Qwen2SelfAttentionResult> {
  validateAttentionInputs(input, sequenceLength, config, weights);

  const hiddenElements = sequenceLength * config.hiddenSize;
  const keyValueElements = sequenceLength * config.keyValueHiddenSize;
  const scoreElements = config.numberOfHeads * sequenceLength * sequenceLength;
  const q = new Float32Array(hiddenElements);
  const k = new Float32Array(keyValueElements);
  const v = new Float32Array(keyValueElements);

  await projectSequenceGpu(runtime, input, sequenceLength, config.hiddenSize, weights.query, q, config.hiddenSize, weights.queryBias);
  await projectSequenceGpu(
    runtime,
    input,
    sequenceLength,
    config.hiddenSize,
    weights.key,
    k,
    config.keyValueHiddenSize,
    weights.keyBias,
  );
  await projectSequenceGpu(
    runtime,
    input,
    sequenceLength,
    config.hiddenSize,
    weights.value,
    v,
    config.keyValueHiddenSize,
    weights.valueBias,
  );
  await applyRoPEToQueryGpu(runtime, q, sequenceLength, config);
  await applyRoPEToKeyGpu(runtime, k, sequenceLength, config);

  const concatenated = await causalGqaHeadOutputsGpu(runtime, {
    q,
    k,
    v,
    sequenceLength,
    config,
  });

  const projected = new Float32Array(hiddenElements);
  await projectSequenceGpu(
    runtime,
    concatenated,
    sequenceLength,
    config.hiddenSize,
    weights.output,
    projected,
    config.hiddenSize,
  );

  const rawScores = new Float32Array(scoreElements);
  const scaledScores = new Float32Array(scoreElements);
  const probabilities = new Float32Array(scoreElements);
  rawScores.fill(Number.NaN);
  scaledScores.fill(Number.NaN);
  probabilities.fill(Number.NaN);

  return {
    output: projected,
    debug: {
      q,
      k,
      v,
      rawScores,
      scaledScores,
      probabilities,
      headOutput: concatenated,
      concatenated,
      projected,
    },
  };
}

export function qwen2AttentionWeights(weights: Qwen2AttentionWeights): Qwen2SelfAttentionWeights {
  return {
    query: weights.qProjWeight,
    queryBias: weights.qProjBias,
    key: weights.kProjWeight,
    keyBias: weights.kProjBias,
    value: weights.vProjWeight,
    valueBias: weights.vProjBias,
    output: weights.outProjWeight,
  };
}

function projectSequence(
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  weight: QwenTensorView,
  output: Float32Array,
  outputSize: number,
  bias?: TensorView | Float32Array,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    matrixVectorMultiplyQwen(weight, input, output, {
      bias,
      inputOffset: position * hiddenSize,
      outputOffset: position * outputSize,
    });
  }
}

async function projectSequenceGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  weight: QwenTensorView,
  output: Float32Array,
  outputSize: number,
  bias?: TensorView | Float32Array,
): Promise<void> {
  for (let position = 0; position < sequenceLength; position += 1) {
    const projected = await matrixVectorMultiplyQwenGpu(runtime, weight, input, {
      bias,
      inputOffset: position * hiddenSize,
    });
    output.set(projected, position * outputSize);
  }
}

function writeProjectedKvToCache(
  keys: Float32Array,
  values: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  cache: Qwen2LayerKvCache,
): void {
  validateCacheBuffers(cache, config);
  const elements = sequenceLength * config.keyValueHiddenSize;
  if (elements > cache.keys.length || elements > cache.values.length) {
    throw new Error(
      `cache cannot store ${sequenceLength} positions with keyValueHiddenSize ${config.keyValueHiddenSize}`,
    );
  }

  cache.keys.set(keys.subarray(0, elements), 0);
  cache.values.set(values.subarray(0, elements), 0);
  cache.length = sequenceLength;
}

function validateCacheBuffers(
  cache: Qwen2LayerKvCache,
  config: Qwen2SelfAttentionConfig,
): void {
  if (cache.keys.length !== cache.values.length) {
    throw new Error("cache key/value buffers must have the same length");
  }
  if (cache.keys.length < config.keyValueHiddenSize) {
    throw new Error("cache key/value buffers are too small");
  }
  if (cache.keys.length % config.keyValueHiddenSize !== 0) {
    throw new Error(
      `cache length ${cache.keys.length} must be a multiple of keyValueHiddenSize ${config.keyValueHiddenSize}`,
    );
  }
}

function applyRoPEToQuery(
  q: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  basePosition = 0,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    for (let head = 0; head < config.numberOfHeads; head += 1) {
      const offset = hiddenIndex(position, head, 0, config);
      applyRoPE(q, q, {
        position: basePosition + position,
        theta: config.ropeTheta,
        inputOffset: offset,
        outputOffset: offset,
        headDimension: config.headDimension,
      });
    }
  }
}

async function applyRoPEToQueryGpu(
  runtime: WebGpuRuntime,
  q: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  basePosition = 0,
): Promise<void> {
  for (let position = 0; position < sequenceLength; position += 1) {
    for (let head = 0; head < config.numberOfHeads; head += 1) {
      const offset = hiddenIndex(position, head, 0, config);
      const rotated = await applyRoPEGpu(runtime, q, {
        position: basePosition + position,
        theta: config.ropeTheta,
        inputOffset: offset,
        headDimension: config.headDimension,
      });
      q.set(rotated, offset);
    }
  }
}

function applyRoPEToKey(
  k: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  basePosition = 0,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    for (let keyValueHead = 0; keyValueHead < config.numberOfKeyValueHeads; keyValueHead += 1) {
      const offset = kvHiddenIndex(position, keyValueHead, 0, config);
      applyRoPE(k, k, {
        position: basePosition + position,
        theta: config.ropeTheta,
        inputOffset: offset,
        outputOffset: offset,
        headDimension: config.headDimension,
      });
    }
  }
}

async function applyRoPEToKeyGpu(
  runtime: WebGpuRuntime,
  k: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  basePosition = 0,
): Promise<void> {
  for (let position = 0; position < sequenceLength; position += 1) {
    for (let keyValueHead = 0; keyValueHead < config.numberOfKeyValueHeads; keyValueHead += 1) {
      const offset = kvHiddenIndex(position, keyValueHead, 0, config);
      const rotated = await applyRoPEGpu(runtime, k, {
        position: basePosition + position,
        theta: config.ropeTheta,
        inputOffset: offset,
        headDimension: config.headDimension,
      });
      k.set(rotated, offset);
    }
  }
}

async function causalGqaHeadOutputsGpu(
  runtime: WebGpuRuntime,
  options: {
    q: Float32Array;
    k: Float32Array;
    v: Float32Array;
    sequenceLength: number;
    config: Qwen2SelfAttentionConfig;
    outputLastPositionOnly?: boolean;
  },
): Promise<Float32Array> {
  const outputSequenceLength = options.outputLastPositionOnly ? 1 : options.sequenceLength;
  const outputLength = outputSequenceLength * options.config.hiddenSize;
  const qBuffer = createStorageBuffer(runtime, options.q);
  const kBuffer = createStorageBuffer(runtime, options.k);
  const vBuffer = createStorageBuffer(runtime, options.v);
  const outputBuffer = createStorageBuffer(runtime, outputLength * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([
      options.sequenceLength,
      options.config.hiddenSize,
      options.config.numberOfHeads,
      options.config.numberOfKeyValueHeads,
      options.config.headDimension,
      options.config.keyValueHiddenSize,
      options.outputLastPositionOnly ? 1 : 0,
      0,
    ]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      causalGqaHeadOutputsShader,
      [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(outputLength / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, outputLength);
  } finally {
    destroyBuffers(qBuffer, kBuffer, vBuffer, outputBuffer, paramsBuffer);
  }
}

const causalGqaHeadOutputsShader = `
struct Params {
  sequenceLength: u32,
  hiddenSize: u32,
  numberOfHeads: u32,
  numberOfKeyValueHeads: u32,
  headDimension: u32,
  keyValueHiddenSize: u32,
  outputLastPositionOnly: u32,
  _padding: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn hiddenIndex(position: u32, head: u32, dimension: u32) -> u32 {
  return position * params.hiddenSize + head * params.headDimension + dimension;
}

fn kvHiddenIndex(position: u32, keyValueHead: u32, dimension: u32) -> u32 {
  return position * params.keyValueHiddenSize + keyValueHead * params.headDimension + dimension;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let outputIndex = globalId.x;
  let outputSequenceLength = select(params.sequenceLength, 1u, params.outputLastPositionOnly == 1u);
  let outputLength = outputSequenceLength * params.hiddenSize;
  if (outputIndex >= outputLength) {
    return;
  }

  let outputPosition = outputIndex / params.hiddenSize;
  let queryPosition = select(outputPosition, params.sequenceLength - 1u, params.outputLastPositionOnly == 1u);
  let hiddenOffset = outputIndex % params.hiddenSize;
  let head = hiddenOffset / params.headDimension;
  let dimension = hiddenOffset % params.headDimension;
  let groupSize = params.numberOfHeads / params.numberOfKeyValueHeads;
  let keyValueHead = head / groupSize;
  let scale = inverseSqrt(f32(params.headDimension));

  var maxScore = -3.4028234663852886e38;
  for (var keyPosition = 0u; keyPosition <= queryPosition; keyPosition = keyPosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[hiddenIndex(queryPosition, head, scoreDimension)] *
        k[kvHiddenIndex(keyPosition, keyValueHead, scoreDimension)];
    }
    maxScore = max(maxScore, score * scale);
  }

  var sum = 0.0;
  var weightedValue = 0.0;
  for (var valuePosition = 0u; valuePosition <= queryPosition; valuePosition = valuePosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[hiddenIndex(queryPosition, head, scoreDimension)] *
        k[kvHiddenIndex(valuePosition, keyValueHead, scoreDimension)];
    }
    let probabilityNumerator = exp(score * scale - maxScore);
    sum = sum + probabilityNumerator;
    weightedValue = weightedValue + probabilityNumerator * v[kvHiddenIndex(valuePosition, keyValueHead, dimension)];
  }

  output[outputIndex] = weightedValue / sum;
}
`;

function computeCausalGqaHeadOutputs(options: {
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  rawScores: Float32Array;
  scaledScores: Float32Array;
  probabilities: Float32Array;
  headOutput: Float32Array;
  concatenated: Float32Array;
  sequenceLength: number;
  config: Qwen2SelfAttentionConfig;
}): void {
  const {
    q,
    k,
    v,
    rawScores,
    scaledScores,
    probabilities,
    headOutput,
    concatenated,
    sequenceLength,
    config,
  } = options;
  const scale = 1 / Math.sqrt(config.headDimension);

  for (let head = 0; head < config.numberOfHeads; head += 1) {
    const keyValueHead = keyValueHeadForQueryHead(head, config);
    for (let queryPosition = 0; queryPosition < sequenceLength; queryPosition += 1) {
      const visibleLength = queryPosition + 1;
      const scoreOffset = scoreIndex(head, queryPosition, 0, sequenceLength);

      for (let keyPosition = 0; keyPosition <= queryPosition; keyPosition += 1) {
        let score = 0;
        for (let dimension = 0; dimension < config.headDimension; dimension += 1) {
          score +=
            (q[hiddenIndex(queryPosition, head, dimension, config)] ?? 0) *
            (k[kvHiddenIndex(keyPosition, keyValueHead, dimension, config)] ?? 0);
        }

        const index = scoreOffset + keyPosition;
        rawScores[index] = score;
        scaledScores[index] = score * scale;
      }

      softmax(scaledScores, probabilities, {
        inputOffset: scoreOffset,
        outputOffset: scoreOffset,
        length: visibleLength,
      });

      for (let dimension = 0; dimension < config.headDimension; dimension += 1) {
        let value = 0;
        for (let valuePosition = 0; valuePosition <= queryPosition; valuePosition += 1) {
          value +=
            (probabilities[scoreOffset + valuePosition] ?? 0) *
            (v[kvHiddenIndex(valuePosition, keyValueHead, dimension, config)] ?? 0);
        }

        const outputIndex = hiddenIndex(queryPosition, head, dimension, config);
        headOutput[outputIndex] = value;
        concatenated[outputIndex] = value;
      }
    }
  }
}

function validateAttentionInputs(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
): void {
  validateSequenceInput(input, sequenceLength, config.hiddenSize);
  validateAttentionConfig(config);
  validateAttentionWeights(config, weights);
}

function validateAttentionConfig(config: Qwen2SelfAttentionConfig): void {
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.numberOfHeads, "numberOfHeads");
  assertPositiveInteger(config.numberOfKeyValueHeads, "numberOfKeyValueHeads");
  assertPositiveInteger(config.headDimension, "headDimension");
  assertPositiveInteger(config.keyValueHiddenSize, "keyValueHiddenSize");
  assertPositiveNumber(config.ropeTheta, "ropeTheta");

  if (config.hiddenSize !== config.numberOfHeads * config.headDimension) {
    throw new Error("Qwen2 hiddenSize must equal numberOfHeads * headDimension");
  }
  if (config.keyValueHiddenSize !== config.numberOfKeyValueHeads * config.headDimension) {
    throw new Error("Qwen2 keyValueHiddenSize must equal numberOfKeyValueHeads * headDimension");
  }
  if (config.numberOfHeads % config.numberOfKeyValueHeads !== 0) {
    throw new Error("Qwen2 numberOfHeads must be divisible by numberOfKeyValueHeads");
  }
  if (config.headDimension % 2 !== 0) {
    throw new Error(`Qwen2 headDimension must be even for RoPE, got ${config.headDimension}`);
  }
}

function validateAttentionWeights(
  config: Qwen2SelfAttentionConfig,
  weights: Qwen2SelfAttentionWeights,
): void {
  requireMatrixShape(weights.query, "query", config.hiddenSize, config.hiddenSize);
  requireOptionalVectorShape(weights.queryBias, "queryBias", config.hiddenSize);
  requireMatrixShape(weights.key, "key", config.keyValueHiddenSize, config.hiddenSize);
  requireOptionalVectorShape(weights.keyBias, "keyBias", config.keyValueHiddenSize);
  requireMatrixShape(weights.value, "value", config.keyValueHiddenSize, config.hiddenSize);
  requireOptionalVectorShape(weights.valueBias, "valueBias", config.keyValueHiddenSize);
  requireMatrixShape(weights.output, "output", config.hiddenSize, config.hiddenSize);
}

function validateSequenceInput(
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
): void {
  assertPositiveInteger(sequenceLength, "sequenceLength");
  assertPositiveInteger(hiddenSize, "hiddenSize");

  const expectedLength = sequenceLength * hiddenSize;
  if (input.length !== expectedLength) {
    throw new Error(`input length is ${input.length}, expected ${expectedLength}`);
  }
}

function requireMatrixShape(
  tensor: QwenTensorView,
  name: string,
  expectedRows: number,
  expectedColumns: number,
): void {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }

  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows !== expectedRows || columns !== expectedColumns) {
    throw new Error(
      `${name} shape is [${tensor.shape.join(", ")}], expected [${expectedRows}, ${expectedColumns}]`,
    );
  }
}

function requireOptionalVectorShape(
  tensor: TensorView | Float32Array | undefined,
  name: string,
  expectedLength: number,
): void {
  if (!tensor) {
    return;
  }
  const data = tensor instanceof Float32Array ? tensor : tensor.data;
  if (data.length !== expectedLength) {
    throw new Error(`${name} length is ${data.length}, expected ${expectedLength}`);
  }
  if (!(tensor instanceof Float32Array) && (tensor.shape.length !== 1 || tensor.shape[0] !== expectedLength)) {
    throw new Error(`${name} must be rank 1 with length ${expectedLength}`);
  }
}

function keyValueHeadForQueryHead(
  queryHead: number,
  config: Qwen2SelfAttentionConfig,
): number {
  const groupSize = config.numberOfHeads / config.numberOfKeyValueHeads;
  return Math.floor(queryHead / groupSize);
}

function hiddenIndex(
  position: number,
  head: number,
  dimension: number,
  config: Qwen2SelfAttentionConfig,
): number {
  return position * config.hiddenSize + head * config.headDimension + dimension;
}

function kvHiddenIndex(
  position: number,
  keyValueHead: number,
  dimension: number,
  config: Qwen2SelfAttentionConfig,
): number {
  return position * config.keyValueHiddenSize + keyValueHead * config.headDimension + dimension;
}

function scoreIndex(
  head: number,
  queryPosition: number,
  keyPosition: number,
  sequenceLength: number,
): number {
  return head * sequenceLength * sequenceLength + queryPosition * sequenceLength + keyPosition;
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

function assertPositiveNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number, got ${String(value)}`);
  }
}

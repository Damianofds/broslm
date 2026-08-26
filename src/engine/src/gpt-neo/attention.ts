import type { LayerKvCache } from "./attentionCache";
import type {
  AttentionWeights as BoundAttentionWeights,
  LayerNormWeights,
  TensorView,
} from "./loader";
import { layerNorm, layerNormGpu } from "../primitives/layerNorm";
import { matrixVectorMultiply, matrixVectorMultiplyGpu } from "../primitives/matrixVectorMultiply";
import { residualAdd, residualAddGpu } from "../primitives/residualAdd";
import { softmax } from "../primitives/softmax";
import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export interface CausalSelfAttentionConfig {
  hiddenSize: number;
  numberOfHeads: number;
  headDimension: number;
}

export interface CausalSelfAttentionWeights {
  query: TensorView;
  key: TensorView;
  value: TensorView;
  output: TensorView;
  outputBias?: TensorView | Float32Array;
}

export interface CausalSelfAttentionDebug {
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

export interface CausalSelfAttentionResult {
  output: Float32Array;
  debug: CausalSelfAttentionDebug;
}

export interface TransformerAttentionBlockWeights {
  layerNorm: LayerNormWeights;
  attention: BoundAttentionWeights;
}

export function causalSelfAttention(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): Float32Array {
  return causalSelfAttentionWithDebug(input, sequenceLength, config, weights).output;
}

export async function causalSelfAttentionGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): Promise<Float32Array> {
  return (await causalSelfAttentionWithDebugGpu(runtime, input, sequenceLength, config, weights)).output;
}

export function causalSelfAttentionPrefill(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
  cache: LayerKvCache,
): Float32Array {
  const result = causalSelfAttentionWithDebug(input, sequenceLength, config, weights);
  writeProjectedKvToCache(result.debug.k, result.debug.v, sequenceLength, config, cache);
  return result.output;
}

export async function causalSelfAttentionPrefillGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
  cache: LayerKvCache,
): Promise<Float32Array> {
  const result = await causalSelfAttentionWithDebugGpu(runtime, input, sequenceLength, config, weights);
  writeProjectedKvToCache(result.debug.k, result.debug.v, sequenceLength, config, cache);
  return result.output;
}

export function causalSelfAttentionIncremental(
  input: Float32Array,
  position: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
  cache: LayerKvCache,
): Float32Array {
  validateAttentionConfig(config);
  requireMatrixShape(weights.query, "query", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.key, "key", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.value, "value", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.output, "output", config.hiddenSize, config.hiddenSize);
  if (input.length !== config.hiddenSize) {
    throw new Error(`input length is ${input.length}, expected ${config.hiddenSize}`);
  }
  if (cache.length !== position) {
    throw new Error(`cache length is ${cache.length}, expected current position ${position}`);
  }
  validateCacheBuffers(cache, config);

  const { hiddenSize, numberOfHeads, headDimension } = config;
  const q = new Float32Array(hiddenSize);
  const k = new Float32Array(hiddenSize);
  const v = new Float32Array(hiddenSize);
  const concatenated = new Float32Array(hiddenSize);
  const projected = new Float32Array(hiddenSize);
  const scores = new Float32Array(position + 1);
  const probabilities = new Float32Array(position + 1);

  matrixVectorMultiply(weights.query, input, q);
  matrixVectorMultiply(weights.key, input, k);
  matrixVectorMultiply(weights.value, input, v);

  const cacheOffset = position * hiddenSize;
  cache.keys.set(k, cacheOffset);
  cache.values.set(v, cacheOffset);

  const scale = 1 / Math.sqrt(headDimension);
  for (let head = 0; head < numberOfHeads; head += 1) {
    for (let keyPosition = 0; keyPosition <= position; keyPosition += 1) {
      let score = 0;
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        score +=
          (q[head * headDimension + dimension] ?? 0) *
          (cache.keys[hiddenIndex(keyPosition, head, dimension, hiddenSize, headDimension)] ?? 0);
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
          (cache.values[hiddenIndex(valuePosition, head, dimension, hiddenSize, headDimension)] ?? 0);
      }
      concatenated[head * headDimension + dimension] = value;
    }
  }

  matrixVectorMultiply(weights.output, concatenated, projected, {
    bias: weights.outputBias,
  });
  cache.length = position + 1;
  return projected;
}

export async function causalSelfAttentionIncrementalGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  position: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
  cache: LayerKvCache,
): Promise<Float32Array> {
  validateAttentionConfig(config);
  requireMatrixShape(weights.query, "query", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.key, "key", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.value, "value", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.output, "output", config.hiddenSize, config.hiddenSize);
  if (input.length !== config.hiddenSize) {
    throw new Error(`input length is ${input.length}, expected ${config.hiddenSize}`);
  }
  if (cache.length !== position) {
    throw new Error(`cache length is ${cache.length}, expected current position ${position}`);
  }
  validateCacheBuffers(cache, config);

  const q = await matrixVectorMultiplyGpu(runtime, weights.query, input);
  const k = await matrixVectorMultiplyGpu(runtime, weights.key, input);
  const v = await matrixVectorMultiplyGpu(runtime, weights.value, input);

  const cacheOffset = position * config.hiddenSize;
  cache.keys.set(k, cacheOffset);
  cache.values.set(v, cacheOffset);

  const qSequence = new Float32Array((position + 1) * config.hiddenSize);
  qSequence.set(q, position * config.hiddenSize);
  const concatenated = await causalHeadOutputsGpu(runtime, {
    q: qSequence,
    k: cache.keys.subarray(0, (position + 1) * config.hiddenSize),
    v: cache.values.subarray(0, (position + 1) * config.hiddenSize),
    sequenceLength: position + 1,
    hiddenSize: config.hiddenSize,
    numberOfHeads: config.numberOfHeads,
    headDimension: config.headDimension,
    outputLastPositionOnly: true,
  });

  const projected = await matrixVectorMultiplyGpu(runtime, weights.output, concatenated, {
    bias: weights.outputBias,
  });
  cache.length = position + 1;
  return projected;
}

export function causalSelfAttentionWithDebug(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): CausalSelfAttentionResult {
  validateAttentionInputs(input, sequenceLength, config, weights);

  const { hiddenSize, numberOfHeads, headDimension } = config;
  const hiddenElements = sequenceLength * hiddenSize;
  const scoreElements = numberOfHeads * sequenceLength * sequenceLength;
  const q = new Float32Array(hiddenElements);
  const k = new Float32Array(hiddenElements);
  const v = new Float32Array(hiddenElements);
  const rawScores = new Float32Array(scoreElements);
  const scaledScores = new Float32Array(scoreElements);
  const probabilities = new Float32Array(scoreElements);
  const headOutput = new Float32Array(hiddenElements);
  const concatenated = new Float32Array(hiddenElements);
  const projected = new Float32Array(hiddenElements);

  rawScores.fill(Number.NaN);
  scaledScores.fill(Number.NaN);
  probabilities.fill(Number.NaN);

  projectSequence(input, sequenceLength, hiddenSize, weights.query, q);
  projectSequence(input, sequenceLength, hiddenSize, weights.key, k);
  projectSequence(input, sequenceLength, hiddenSize, weights.value, v);

  computeCausalHeadOutputs({
    q,
    k,
    v,
    rawScores,
    scaledScores,
    probabilities,
    headOutput,
    concatenated,
    sequenceLength,
    hiddenSize,
    numberOfHeads,
    headDimension,
  });

  projectSequence(
    concatenated,
    sequenceLength,
    hiddenSize,
    weights.output,
    projected,
    weights.outputBias,
  );

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

export async function causalSelfAttentionWithDebugGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): Promise<CausalSelfAttentionResult> {
  validateAttentionInputs(input, sequenceLength, config, weights);

  const { hiddenSize, numberOfHeads, headDimension } = config;
  const hiddenElements = sequenceLength * hiddenSize;
  const scoreElements = numberOfHeads * sequenceLength * sequenceLength;
  const q = new Float32Array(hiddenElements);
  const k = new Float32Array(hiddenElements);
  const v = new Float32Array(hiddenElements);

  await projectSequenceGpu(runtime, input, sequenceLength, hiddenSize, weights.query, q);
  await projectSequenceGpu(runtime, input, sequenceLength, hiddenSize, weights.key, k);
  await projectSequenceGpu(runtime, input, sequenceLength, hiddenSize, weights.value, v);

  const concatenated = await causalHeadOutputsGpu(runtime, {
    q,
    k,
    v,
    sequenceLength,
    hiddenSize,
    numberOfHeads,
    headDimension,
  });

  const projected = new Float32Array(hiddenElements);
  await projectSequenceGpu(
    runtime,
    concatenated,
    sequenceLength,
    hiddenSize,
    weights.output,
    projected,
    weights.outputBias,
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

export function gptNeoAttentionWeights(weights: BoundAttentionWeights): CausalSelfAttentionWeights {
  return {
    query: weights.qProjWeight,
    key: weights.kProjWeight,
    value: weights.vProjWeight,
    output: weights.outProjWeight,
    outputBias: weights.outProjBias,
  };
}

export function transformerAttentionBlock(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig & { layerNormEpsilon: number },
  weights: TransformerAttentionBlockWeights,
): Float32Array {
  validateSequenceInput(input, sequenceLength, config.hiddenSize);

  const normed = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    layerNorm(input, weights.layerNorm.weight, weights.layerNorm.bias, normed, {
      inputOffset: offset,
      outputOffset: offset,
      featureSize: config.hiddenSize,
      epsilon: config.layerNormEpsilon,
    });
  }

  const attention = causalSelfAttention(
    normed,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
  );
  const output = new Float32Array(input.length);
  residualAdd(attention, input, output);
  return output;
}

export async function transformerAttentionBlockGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig & { layerNormEpsilon: number },
  weights: TransformerAttentionBlockWeights,
): Promise<Float32Array> {
  validateSequenceInput(input, sequenceLength, config.hiddenSize);

  const normed = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenNormed = await layerNormGpu(runtime, input, weights.layerNorm.weight, weights.layerNorm.bias, {
      inputOffset: offset,
      featureSize: config.hiddenSize,
      epsilon: config.layerNormEpsilon,
    });
    normed.set(tokenNormed, offset);
  }

  const attention = await causalSelfAttentionGpu(
    runtime,
    normed,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
  );
  return residualAddGpu(runtime, attention, input);
}

function projectSequence(
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  weight: TensorView,
  output: Float32Array,
  bias?: TensorView | Float32Array,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    matrixVectorMultiply(weight, input, output, {
      bias,
      inputOffset: offset,
      outputOffset: offset,
    });
  }
}

async function projectSequenceGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  weight: TensorView,
  output: Float32Array,
  bias?: TensorView | Float32Array,
): Promise<void> {
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    const projected = await matrixVectorMultiplyGpu(runtime, weight, input, {
      bias,
      inputOffset: offset,
    });
    output.set(projected, offset);
  }
}

async function causalHeadOutputsGpu(
  runtime: WebGpuRuntime,
  options: {
    q: Float32Array;
    k: Float32Array;
    v: Float32Array;
    sequenceLength: number;
    hiddenSize: number;
    numberOfHeads: number;
    headDimension: number;
    outputLastPositionOnly?: boolean;
  },
): Promise<Float32Array> {
  const outputSequenceLength = options.outputLastPositionOnly ? 1 : options.sequenceLength;
  const outputLength = outputSequenceLength * options.hiddenSize;
  const qBuffer = createStorageBuffer(runtime, options.q);
  const kBuffer = createStorageBuffer(runtime, options.k);
  const vBuffer = createStorageBuffer(runtime, options.v);
  const outputBuffer = createStorageBuffer(runtime, outputLength * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([
      options.sequenceLength,
      options.hiddenSize,
      options.numberOfHeads,
      options.headDimension,
      options.outputLastPositionOnly ? 1 : 0,
      0,
      0,
      0,
    ]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      causalHeadOutputsShader,
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

const causalHeadOutputsShader = `
struct Params {
  sequenceLength: u32,
  hiddenSize: u32,
  numberOfHeads: u32,
  headDimension: u32,
  outputLastPositionOnly: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn hiddenIndex(position: u32, head: u32, dimension: u32) -> u32 {
  return position * params.hiddenSize + head * params.headDimension + dimension;
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
  let scale = inverseSqrt(f32(params.headDimension));

  var maxScore = -3.4028234663852886e38;
  for (var keyPosition = 0u; keyPosition <= queryPosition; keyPosition = keyPosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[hiddenIndex(queryPosition, head, scoreDimension)] *
        k[hiddenIndex(keyPosition, head, scoreDimension)];
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
        k[hiddenIndex(valuePosition, head, scoreDimension)];
    }
    let probabilityNumerator = exp(score * scale - maxScore);
    sum = sum + probabilityNumerator;
    weightedValue = weightedValue + probabilityNumerator * v[hiddenIndex(valuePosition, head, dimension)];
  }

  output[outputIndex] = weightedValue / sum;
}
`;

function writeProjectedKvToCache(
  keys: Float32Array,
  values: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  cache: LayerKvCache,
): void {
  validateCacheBuffers(cache, config);
  const elements = sequenceLength * config.hiddenSize;
  if (elements > cache.keys.length || elements > cache.values.length) {
    throw new Error(
      `cache cannot store ${sequenceLength} positions with hiddenSize ${config.hiddenSize}`,
    );
  }

  cache.keys.set(keys.subarray(0, elements), 0);
  cache.values.set(values.subarray(0, elements), 0);
  cache.length = sequenceLength;
}

function validateCacheBuffers(cache: LayerKvCache, config: CausalSelfAttentionConfig): void {
  if (cache.keys.length !== cache.values.length) {
    throw new Error("cache key/value buffers must have the same length");
  }
  if (cache.keys.length < config.hiddenSize) {
    throw new Error("cache key/value buffers are too small");
  }
  if (cache.keys.length % config.hiddenSize !== 0) {
    throw new Error(`cache length ${cache.keys.length} must be a multiple of hiddenSize ${config.hiddenSize}`);
  }
}

function computeCausalHeadOutputs(options: {
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  rawScores: Float32Array;
  scaledScores: Float32Array;
  probabilities: Float32Array;
  headOutput: Float32Array;
  concatenated: Float32Array;
  sequenceLength: number;
  hiddenSize: number;
  numberOfHeads: number;
  headDimension: number;
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
    hiddenSize,
    numberOfHeads,
    headDimension,
  } = options;
  const scale = 1 / Math.sqrt(headDimension);

  for (let head = 0; head < numberOfHeads; head += 1) {
    for (let queryPosition = 0; queryPosition < sequenceLength; queryPosition += 1) {
      const visibleLength = queryPosition + 1;
      const scoreOffset = scoreIndex(head, queryPosition, 0, sequenceLength);

      for (let keyPosition = 0; keyPosition <= queryPosition; keyPosition += 1) {
        let score = 0;
        for (let dimension = 0; dimension < headDimension; dimension += 1) {
          score +=
            (q[hiddenIndex(queryPosition, head, dimension, hiddenSize, headDimension)] ?? 0) *
            (k[hiddenIndex(keyPosition, head, dimension, hiddenSize, headDimension)] ?? 0);
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

      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        let value = 0;
        for (let valuePosition = 0; valuePosition <= queryPosition; valuePosition += 1) {
          value +=
            (probabilities[scoreOffset + valuePosition] ?? 0) *
            (v[hiddenIndex(valuePosition, head, dimension, hiddenSize, headDimension)] ?? 0);
        }

        const outputIndex = hiddenIndex(queryPosition, head, dimension, hiddenSize, headDimension);
        headOutput[outputIndex] = value;
        concatenated[outputIndex] = value;
      }
    }
  }
}

function validateAttentionInputs(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): void {
  validateSequenceInput(input, sequenceLength, config.hiddenSize);
  validateAttentionConfig(config);
  requireMatrixShape(weights.query, "query", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.key, "key", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.value, "value", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.output, "output", config.hiddenSize, config.hiddenSize);

  if (weights.outputBias instanceof Float32Array) {
    if (weights.outputBias.length !== config.hiddenSize) {
      throw new Error(`outputBias length is ${weights.outputBias.length}, expected ${config.hiddenSize}`);
    }
  } else if (weights.outputBias) {
    requireVectorShape(weights.outputBias, "outputBias", config.hiddenSize);
  }
}

function validateAttentionConfig(config: CausalSelfAttentionConfig): void {
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.numberOfHeads, "numberOfHeads");
  assertPositiveInteger(config.headDimension, "headDimension");

  if (config.hiddenSize % config.numberOfHeads !== 0) {
    throw new Error(
      `hiddenSize ${config.hiddenSize} must be divisible by numberOfHeads ${config.numberOfHeads}`,
    );
  }

  const expectedHeadDimension = config.hiddenSize / config.numberOfHeads;
  if (config.headDimension !== expectedHeadDimension) {
    throw new Error(
      `headDimension ${config.headDimension} must equal hiddenSize / numberOfHeads ${expectedHeadDimension}`,
    );
  }
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
  tensor: TensorView,
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
  if (tensor.data.length !== rows * columns) {
    throw new Error(`${name} data length does not match shape [${rows}, ${columns}]`);
  }
}

function requireVectorShape(tensor: TensorView, name: string, expectedLength: number): void {
  if (tensor.shape.length !== 1 || tensor.shape[0] !== expectedLength) {
    throw new Error(`${name} must be rank 1 with length ${expectedLength}`);
  }
  if (tensor.data.length !== expectedLength) {
    throw new Error(`${name} data length does not match shape [${expectedLength}]`);
  }
}

function hiddenIndex(
  position: number,
  head: number,
  dimension: number,
  hiddenSize: number,
  headDimension: number,
): number {
  return position * hiddenSize + head * headDimension + dimension;
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

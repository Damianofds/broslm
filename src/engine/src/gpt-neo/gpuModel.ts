import type { LayerKvCache, ModelKvCache } from "./attentionCache";
import { resetModelKvCache } from "./attentionCache";
import type { LoadedModel, TensorView } from "./loader";
import {
  createStaticStorageBuffer,
  createStorageBuffer,
  destroyBuffers,
  encodeComputeShader,
  readFloat32Buffer,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

interface ResidentPrefillBuffers {
  temps: GPUBuffer[];
  encoder: GPUCommandEncoder;
  dummyBias: GPUBuffer;
}

export async function gptNeoPrefillLogitsResidentGpu(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  resetModelKvCache(cache);

  const sequenceLength = inputIds.length;
  const hiddenSize = model.config.hiddenSize;
  const hiddenElements = sequenceLength * hiddenSize;
  const intermediateElements = sequenceLength * model.config.intermediateSize;
  const context = createResidentPrefillBuffers(runtime);

  const tokenIdsBuffer = tempBuffer(context, runtime, Uint32Array.from(inputIds));
  const hiddenA = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const hiddenB = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const normed = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const q = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const attentionOutput = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const mlpIntermediate = tempBuffer(
    context,
    runtime,
    intermediateElements * Float32Array.BYTES_PER_ELEMENT,
  );
  const mlpActivated = tempBuffer(
    context,
    runtime,
    intermediateElements * Float32Array.BYTES_PER_ELEMENT,
  );
  const mlpOutput = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const finalHidden = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const logits = tempBuffer(
    context,
    runtime,
    model.config.vocabularySize * Float32Array.BYTES_PER_ELEMENT,
  );

  const layerKeyBuffers: GPUBuffer[] = [];
  const layerValueBuffers: GPUBuffer[] = [];

  encodeGptNeoEmbedding(context, runtime, {
    tokenIds: tokenIdsBuffer,
    tokenEmbedding: tensorBuffer(runtime, model.weights.tokenEmbedding),
    positionEmbedding: tensorBuffer(runtime, model.weights.positionEmbedding),
    output: hiddenA,
    sequenceLength,
    hiddenSize,
  });

  let layerInput = hiddenA;
  let layerOutput = hiddenB;

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    if (!layerWeights) {
      throw new Error(`missing layer at index ${layerIndex}`);
    }

    encodeLayerNorm(context, runtime, {
      input: layerInput,
      weight: tensorBuffer(runtime, layerWeights.ln1.weight),
      bias: tensorBuffer(runtime, layerWeights.ln1.bias),
      output: normed,
      sequenceLength,
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    });

    const layerKeys = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
    const layerValues = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
    layerKeyBuffers.push(layerKeys);
    layerValueBuffers.push(layerValues);

    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.qProjWeight),
      output: q,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.kProjWeight),
      output: layerKeys,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.vProjWeight),
      output: layerValues,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeCausalAttention(context, runtime, {
      q,
      k: layerKeys,
      v: layerValues,
      output: attentionOutput,
      sequenceLength,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      headDimension: model.config.headDimension,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: attentionOutput,
      weight: tensorBuffer(runtime, layerWeights.attention.outProjWeight),
      bias: tensorBuffer(runtime, layerWeights.attention.outProjBias),
      output: layerOutput,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeResidualAdd(context, runtime, {
      left: layerOutput,
      right: layerInput,
      output: attentionOutput,
      length: hiddenElements,
    });

    encodeLayerNorm(context, runtime, {
      input: attentionOutput,
      weight: tensorBuffer(runtime, layerWeights.ln2.weight),
      bias: tensorBuffer(runtime, layerWeights.ln2.bias),
      output: normed,
      sequenceLength,
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.mlp.cFcWeight),
      bias: tensorBuffer(runtime, layerWeights.mlp.cFcBias),
      output: mlpIntermediate,
      inputSize: hiddenSize,
      outputSize: model.config.intermediateSize,
      sequenceLength,
    });
    encodeGelu(context, runtime, {
      input: mlpIntermediate,
      output: mlpActivated,
      length: intermediateElements,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: mlpActivated,
      weight: tensorBuffer(runtime, layerWeights.mlp.cProjWeight),
      bias: tensorBuffer(runtime, layerWeights.mlp.cProjBias),
      output: mlpOutput,
      inputSize: model.config.intermediateSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeResidualAdd(context, runtime, {
      left: mlpOutput,
      right: attentionOutput,
      output: layerInput,
      length: hiddenElements,
    });
  }

  encodeLayerNorm(context, runtime, {
    input: layerInput,
    inputOffset: (sequenceLength - 1) * hiddenSize,
    weight: tensorBuffer(runtime, model.weights.finalLayerNorm.weight),
    bias: tensorBuffer(runtime, model.weights.finalLayerNorm.bias),
    output: finalHidden,
    sequenceLength: 1,
    featureSize: hiddenSize,
    epsilon: model.config.layerNormEpsilon,
  });
  encodeMatrixMultiplySequence(context, runtime, {
    input: finalHidden,
    weight: tensorBuffer(runtime, model.weights.lmHead),
    output: logits,
    inputSize: hiddenSize,
    outputSize: model.config.vocabularySize,
    sequenceLength: 1,
  });

  runtime.device.queue.submit([context.encoder.finish()]);
  const logitsCpu = await readFloat32Buffer(runtime, logits, model.config.vocabularySize);

  for (let layerIndex = 0; layerIndex < cache.layers.length; layerIndex += 1) {
    const layerCache = cache.layers[layerIndex];
    const layerKeys = layerKeyBuffers[layerIndex];
    const layerValues = layerValueBuffers[layerIndex];
    if (!layerCache || !layerKeys || !layerValues) {
      throw new Error(`missing GPU/CPU cache at layer ${layerIndex}`);
    }
    await readLayerCache(runtime, layerKeys, layerValues, sequenceLength, model.config.hiddenSize, layerCache);
  }

  cache.inputIds.push(...inputIds);
  destroyBuffers(...context.temps);
  return logitsCpu;
}

function createResidentPrefillBuffers(runtime: WebGpuRuntime): ResidentPrefillBuffers {
  const dummyBias = createStorageBuffer(runtime, Float32Array.of(0));
  return {
    temps: [dummyBias],
    encoder: runtime.device.createCommandEncoder(),
    dummyBias,
  };
}

function tempBuffer(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  dataOrByteLength: Float32Array | Uint32Array | Uint8Array | ArrayBuffer | number,
  usage?: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = createStorageBuffer(runtime, dataOrByteLength, usage);
  context.temps.push(buffer);
  return buffer;
}

function tensorBuffer(runtime: WebGpuRuntime, tensor: TensorView): GPUBuffer {
  return createStaticStorageBuffer(runtime, tensor.data);
}

function paramsBuffer(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  data: Float32Array | Uint32Array,
): GPUBuffer {
  return tempBuffer(context, runtime, data, webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst);
}

function encodeGptNeoEmbedding(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    tokenIds: GPUBuffer;
    tokenEmbedding: GPUBuffer;
    positionEmbedding: GPUBuffer;
    output: GPUBuffer;
    sequenceLength: number;
    hiddenSize: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([options.sequenceLength, options.hiddenSize, 0, 0]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    gptNeoEmbeddingShader,
    [
      { binding: 0, resource: { buffer: options.tokenIds } },
      { binding: 1, resource: { buffer: options.tokenEmbedding } },
      { binding: 2, resource: { buffer: options.positionEmbedding } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [Math.ceil((options.sequenceLength * options.hiddenSize) / 128)],
  );
}

function encodeLayerNorm(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputOffset?: number;
    weight: GPUBuffer;
    bias: GPUBuffer;
    output: GPUBuffer;
    sequenceLength: number;
    featureSize: number;
    epsilon: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Float32Array([
      options.epsilon,
      options.inputOffset ?? 0,
      options.featureSize,
      options.sequenceLength,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    layerNormSequenceShader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: { buffer: options.weight } },
      { binding: 2, resource: { buffer: options.bias } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.sequenceLength / 64)],
  );
}

function encodeMatrixMultiplySequence(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputBaseOffset?: number;
    weight: GPUBuffer;
    bias?: GPUBuffer;
    output: GPUBuffer;
    inputSize: number;
    outputSize: number;
    sequenceLength: number;
  },
): void {
  const hasBias = options.bias ? 1 : 0;
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.inputSize,
      options.outputSize,
      options.sequenceLength,
      options.inputBaseOffset ?? 0,
      hasBias,
      0,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    matrixMultiplySequenceShader,
    [
      { binding: 0, resource: { buffer: options.weight } },
      { binding: 1, resource: { buffer: options.input } },
      { binding: 2, resource: { buffer: options.bias ?? context.dummyBias } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [Math.ceil((options.sequenceLength * options.outputSize) / 64)],
  );
}

function encodeCausalAttention(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    q: GPUBuffer;
    k: GPUBuffer;
    v: GPUBuffer;
    output: GPUBuffer;
    sequenceLength: number;
    hiddenSize: number;
    numberOfHeads: number;
    headDimension: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.sequenceLength,
      options.hiddenSize,
      options.numberOfHeads,
      options.headDimension,
      0,
      0,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    causalAttentionShader,
    [
      { binding: 0, resource: { buffer: options.q } },
      { binding: 1, resource: { buffer: options.k } },
      { binding: 2, resource: { buffer: options.v } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [Math.ceil((options.sequenceLength * options.hiddenSize) / 128)],
  );
}

function encodeResidualAdd(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    left: GPUBuffer;
    right: GPUBuffer;
    output: GPUBuffer;
    length: number;
  },
): void {
  const params = paramsBuffer(context, runtime, new Uint32Array([options.length, 0, 0, 0]));
  encodeComputeShader(
    runtime,
    context.encoder,
    residualAddShader,
    [
      { binding: 0, resource: { buffer: options.left } },
      { binding: 1, resource: { buffer: options.right } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function encodeGelu(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    output: GPUBuffer;
    length: number;
  },
): void {
  const params = paramsBuffer(context, runtime, new Uint32Array([options.length, 0, 0, 0]));
  encodeComputeShader(
    runtime,
    context.encoder,
    geluShader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

async function readLayerCache(
  runtime: WebGpuRuntime,
  keys: GPUBuffer,
  values: GPUBuffer,
  sequenceLength: number,
  hiddenSize: number,
  cache: LayerKvCache,
): Promise<void> {
  const elementCount = sequenceLength * hiddenSize;
  cache.keys.set(await readFloat32Buffer(runtime, keys, elementCount), 0);
  cache.values.set(await readFloat32Buffer(runtime, values, elementCount), 0);
  cache.length = sequenceLength;
}

const gptNeoEmbeddingShader = `
struct Params {
  sequenceLength: u32,
  hiddenSize: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(1) var<storage, read> tokenEmbedding: array<f32>;
@group(0) @binding(2) var<storage, read> positionEmbedding: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let elementCount = params.sequenceLength * params.hiddenSize;
  if (index >= elementCount) {
    return;
  }
  let position = index / params.hiddenSize;
  let dimension = index % params.hiddenSize;
  let tokenId = tokenIds[position];
  output[index] =
    tokenEmbedding[tokenId * params.hiddenSize + dimension] +
    positionEmbedding[position * params.hiddenSize + dimension];
}
`;

const layerNormSequenceShader = `
struct Params {
  epsilon: f32,
  inputOffset: f32,
  featureSize: f32,
  sequenceLength: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let position = globalId.x;
  let sequenceLength = u32(params.sequenceLength);
  let featureSize = u32(params.featureSize);
  if (position >= sequenceLength) {
    return;
  }

  let inputBase = u32(params.inputOffset) + position * featureSize;
  let outputBase = position * featureSize;
  var mean = 0.0;
  for (var index = 0u; index < featureSize; index = index + 1u) {
    mean = mean + input[inputBase + index];
  }
  mean = mean / f32(featureSize);

  var variance = 0.0;
  for (var index = 0u; index < featureSize; index = index + 1u) {
    let centered = input[inputBase + index] - mean;
    variance = variance + centered * centered;
  }
  variance = variance / f32(featureSize);
  let scale = inverseSqrt(variance + params.epsilon);

  for (var index = 0u; index < featureSize; index = index + 1u) {
    let normalized = (input[inputBase + index] - mean) * scale;
    output[outputBase + index] = normalized * weight[index] + bias[index];
  }
}
`;

const matrixMultiplySequenceShader = `
struct Params {
  inputSize: u32,
  outputSize: u32,
  sequenceLength: u32,
  inputBaseOffset: u32,
  hasBias: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let outputElements = params.sequenceLength * params.outputSize;
  if (index >= outputElements) {
    return;
  }
  let position = index / params.outputSize;
  let row = index % params.outputSize;
  var sum = select(0.0, bias[row], params.hasBias == 1u);
  let weightOffset = row * params.inputSize;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;
  for (var column = 0u; column < params.inputSize; column = column + 1u) {
    sum = sum + weight[weightOffset + column] * input[inputOffset + column];
  }
  output[index] = sum;
}
`;

const causalAttentionShader = `
struct Params {
  sequenceLength: u32,
  hiddenSize: u32,
  numberOfHeads: u32,
  headDimension: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
  _padding3: u32,
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
  let outputLength = params.sequenceLength * params.hiddenSize;
  if (outputIndex >= outputLength) {
    return;
  }

  let queryPosition = outputIndex / params.hiddenSize;
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

const residualAddShader = `
struct Params {
  length: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  output[index] = left[index] + right[index];
}
`;

const geluShader = `
struct Params {
  length: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  let value = input[index];
  output[index] = 0.5 * value * (1.0 + tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)));
}
`;

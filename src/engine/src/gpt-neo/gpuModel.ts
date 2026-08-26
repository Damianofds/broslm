import type { ModelKvCache } from "./attentionCache";
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

interface ResidentGpuLayerCache {
  keys: GPUBuffer;
  values: GPUBuffer;
  length: number;
}

interface ResidentGpuModelCache {
  runtime: WebGpuRuntime;
  maximumSequenceLength: number;
  hiddenSize: number;
  inputIds: number[];
  layers: ResidentGpuLayerCache[];
}

const residentGpuCacheByCpuCache = new WeakMap<ModelKvCache, ResidentGpuModelCache>();

export async function gptNeoPrefillLogitsResidentGpu(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  resetModelKvCache(cache);
  const gpuCache = resetResidentGpuCache(model, cache, runtime);

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

    const layerCache = gpuCache.layers[layerIndex];
    if (!layerCache) {
      throw new Error(`missing GPU cache at layer ${layerIndex}`);
    }

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
      output: layerCache.keys,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.vProjWeight),
      output: layerCache.values,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeCausalAttention(context, runtime, {
      q,
      k: layerCache.keys,
      v: layerCache.values,
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
    const gpuLayerCache = gpuCache.layers[layerIndex];
    if (!layerCache || !gpuLayerCache) {
      throw new Error(`missing CPU/GPU cache at layer ${layerIndex}`);
    }
    layerCache.length = sequenceLength;
    gpuLayerCache.length = sequenceLength;
  }

  cache.inputIds.push(...inputIds);
  gpuCache.inputIds.push(...inputIds);
  destroyBuffers(...context.temps);
  return logitsCpu;
}

export async function gptNeoDecodeTokenLogitsResidentGpu(
  model: LoadedModel,
  tokenId: number,
  position: number,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const gpuCache = requireResidentGpuCache(model, cache, runtime);
  if (cache.inputIds.length !== position) {
    throw new Error(`cache input length is ${cache.inputIds.length}, expected position ${position}`);
  }

  const hiddenSize = model.config.hiddenSize;
  const intermediateSize = model.config.intermediateSize;
  const context = createResidentPrefillBuffers(runtime);
  const hiddenA = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const hiddenB = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const normed = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const q = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const k = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const v = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const attentionOutput = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const mlpIntermediate = tempBuffer(context, runtime, intermediateSize * Float32Array.BYTES_PER_ELEMENT);
  const mlpActivated = tempBuffer(context, runtime, intermediateSize * Float32Array.BYTES_PER_ELEMENT);
  const mlpOutput = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const finalHidden = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const logits = tempBuffer(
    context,
    runtime,
    model.config.vocabularySize * Float32Array.BYTES_PER_ELEMENT,
  );

  encodeGptNeoTokenEmbedding(context, runtime, {
    tokenId,
    position,
    tokenEmbedding: tensorBuffer(runtime, model.weights.tokenEmbedding),
    positionEmbedding: tensorBuffer(runtime, model.weights.positionEmbedding),
    output: hiddenA,
    hiddenSize,
  });

  let layerInput = hiddenA;
  let layerOutput = hiddenB;

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const gpuLayerCache = gpuCache.layers[layerIndex];
    const cpuLayerCache = cache.layers[layerIndex];
    if (!layerWeights || !gpuLayerCache || !cpuLayerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    if (gpuLayerCache.length !== position || cpuLayerCache.length !== position) {
      throw new Error(
        `cache length mismatch at layer ${layerIndex}: GPU ${gpuLayerCache.length}, ` +
          `CPU ${cpuLayerCache.length}, expected ${position}`,
      );
    }

    encodeLayerNorm(context, runtime, {
      input: layerInput,
      weight: tensorBuffer(runtime, layerWeights.ln1.weight),
      bias: tensorBuffer(runtime, layerWeights.ln1.bias),
      output: normed,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.qProjWeight),
      output: q,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.kProjWeight),
      output: k,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.attention.vProjWeight),
      output: v,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });

    const cacheByteOffset = position * hiddenSize * Float32Array.BYTES_PER_ELEMENT;
    const tokenByteLength = hiddenSize * Float32Array.BYTES_PER_ELEMENT;
    context.encoder.copyBufferToBuffer(k, 0, gpuLayerCache.keys, cacheByteOffset, tokenByteLength);
    context.encoder.copyBufferToBuffer(v, 0, gpuLayerCache.values, cacheByteOffset, tokenByteLength);

    encodeCausalAttentionDecode(context, runtime, {
      q,
      k: gpuLayerCache.keys,
      v: gpuLayerCache.values,
      output: attentionOutput,
      position,
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
      sequenceLength: 1,
    });
    encodeResidualAdd(context, runtime, {
      left: layerOutput,
      right: layerInput,
      output: attentionOutput,
      length: hiddenSize,
    });

    encodeLayerNorm(context, runtime, {
      input: attentionOutput,
      weight: tensorBuffer(runtime, layerWeights.ln2.weight),
      bias: tensorBuffer(runtime, layerWeights.ln2.bias),
      output: normed,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: tensorBuffer(runtime, layerWeights.mlp.cFcWeight),
      bias: tensorBuffer(runtime, layerWeights.mlp.cFcBias),
      output: mlpIntermediate,
      inputSize: hiddenSize,
      outputSize: intermediateSize,
      sequenceLength: 1,
    });
    encodeGelu(context, runtime, {
      input: mlpIntermediate,
      output: mlpActivated,
      length: intermediateSize,
    });
    encodeMatrixMultiplySequence(context, runtime, {
      input: mlpActivated,
      weight: tensorBuffer(runtime, layerWeights.mlp.cProjWeight),
      bias: tensorBuffer(runtime, layerWeights.mlp.cProjBias),
      output: mlpOutput,
      inputSize: intermediateSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });
    encodeResidualAdd(context, runtime, {
      left: mlpOutput,
      right: attentionOutput,
      output: layerInput,
      length: hiddenSize,
    });
  }

  encodeLayerNorm(context, runtime, {
    input: layerInput,
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
    const cpuLayerCache = cache.layers[layerIndex];
    const gpuLayerCache = gpuCache.layers[layerIndex];
    if (!cpuLayerCache || !gpuLayerCache) {
      throw new Error(`missing CPU/GPU cache at layer ${layerIndex}`);
    }
    cpuLayerCache.length = position + 1;
    gpuLayerCache.length = position + 1;
  }
  gpuCache.inputIds.push(tokenId);
  destroyBuffers(...context.temps);
  return logitsCpu;
}

export function hasGptNeoResidentGpuCache(
  model: LoadedModel,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): boolean {
  const gpuCache = residentGpuCacheByCpuCache.get(cache);
  return Boolean(gpuCache && residentGpuCacheMatches(model, cache, runtime, gpuCache));
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

function getOrCreateResidentGpuCache(
  model: LoadedModel,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentGpuModelCache {
  const existing = residentGpuCacheByCpuCache.get(cache);
  if (existing && residentGpuCacheShapeMatches(model, runtime, existing)) {
    return existing;
  }
  if (existing) {
    destroyResidentGpuCache(existing);
  }

  const layerByteLength =
    model.config.maximumSequenceLength * model.config.hiddenSize * Float32Array.BYTES_PER_ELEMENT;
  const gpuCache: ResidentGpuModelCache = {
    runtime,
    maximumSequenceLength: model.config.maximumSequenceLength,
    hiddenSize: model.config.hiddenSize,
    inputIds: [],
    layers: Array.from({ length: model.config.numberOfLayers }, () => ({
      keys: createStorageBuffer(runtime, layerByteLength),
      values: createStorageBuffer(runtime, layerByteLength),
      length: 0,
    })),
  };
  residentGpuCacheByCpuCache.set(cache, gpuCache);
  return gpuCache;
}

function resetResidentGpuCache(
  model: LoadedModel,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentGpuModelCache {
  const gpuCache = getOrCreateResidentGpuCache(model, cache, runtime);
  gpuCache.inputIds.length = 0;
  for (const layer of gpuCache.layers) {
    layer.length = 0;
  }
  return gpuCache;
}

function requireResidentGpuCache(
  model: LoadedModel,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentGpuModelCache {
  const gpuCache = residentGpuCacheByCpuCache.get(cache);
  if (!gpuCache || !residentGpuCacheMatches(model, cache, runtime, gpuCache)) {
    throw new Error("GPT-Neo WebGPU decode requires a resident GPU cache initialized by prefill.");
  }
  return gpuCache;
}

function residentGpuCacheMatches(
  model: LoadedModel,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
  gpuCache: ResidentGpuModelCache,
): boolean {
  if (!residentGpuCacheShapeMatches(model, runtime, gpuCache)) {
    return false;
  }
  if (gpuCache.inputIds.length !== cache.inputIds.length) {
    return false;
  }
  for (let index = 0; index < cache.inputIds.length; index += 1) {
    if (gpuCache.inputIds[index] !== cache.inputIds[index]) {
      return false;
    }
  }
  for (let layerIndex = 0; layerIndex < cache.layers.length; layerIndex += 1) {
    const cpuLayer = cache.layers[layerIndex];
    const gpuLayer = gpuCache.layers[layerIndex];
    if (!cpuLayer || !gpuLayer || cpuLayer.length !== gpuLayer.length) {
      return false;
    }
  }
  return true;
}

function residentGpuCacheShapeMatches(
  model: LoadedModel,
  runtime: WebGpuRuntime,
  gpuCache: ResidentGpuModelCache,
): boolean {
  return (
    gpuCache.runtime === runtime &&
    gpuCache.maximumSequenceLength === model.config.maximumSequenceLength &&
    gpuCache.hiddenSize === model.config.hiddenSize &&
    gpuCache.layers.length === model.config.numberOfLayers
  );
}

function destroyResidentGpuCache(gpuCache: ResidentGpuModelCache): void {
  for (const layer of gpuCache.layers) {
    destroyBuffers(layer.keys, layer.values);
  }
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

function encodeGptNeoTokenEmbedding(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    tokenId: number;
    position: number;
    tokenEmbedding: GPUBuffer;
    positionEmbedding: GPUBuffer;
    output: GPUBuffer;
    hiddenSize: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([options.tokenId, options.position, options.hiddenSize, 0]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    gptNeoTokenEmbeddingShader,
    [
      { binding: 0, resource: { buffer: options.tokenEmbedding } },
      { binding: 1, resource: { buffer: options.positionEmbedding } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.hiddenSize / 128)],
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

function encodeCausalAttentionDecode(
  context: ResidentPrefillBuffers,
  runtime: WebGpuRuntime,
  options: {
    q: GPUBuffer;
    k: GPUBuffer;
    v: GPUBuffer;
    output: GPUBuffer;
    position: number;
    hiddenSize: number;
    numberOfHeads: number;
    headDimension: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.position,
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
    causalAttentionDecodeShader,
    [
      { binding: 0, resource: { buffer: options.q } },
      { binding: 1, resource: { buffer: options.k } },
      { binding: 2, resource: { buffer: options.v } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.hiddenSize / 128)],
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

const gptNeoTokenEmbeddingShader = `
struct Params {
  tokenId: u32,
  position: u32,
  hiddenSize: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> tokenEmbedding: array<f32>;
@group(0) @binding(1) var<storage, read> positionEmbedding: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let dimension = globalId.x;
  if (dimension >= params.hiddenSize) {
    return;
  }
  output[dimension] =
    tokenEmbedding[params.tokenId * params.hiddenSize + dimension] +
    positionEmbedding[params.position * params.hiddenSize + dimension];
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

const causalAttentionDecodeShader = `
struct Params {
  position: u32,
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

fn cacheHiddenIndex(position: u32, head: u32, dimension: u32) -> u32 {
  return position * params.hiddenSize + head * params.headDimension + dimension;
}

fn tokenHiddenIndex(head: u32, dimension: u32) -> u32 {
  return head * params.headDimension + dimension;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let outputIndex = globalId.x;
  if (outputIndex >= params.hiddenSize) {
    return;
  }

  let head = outputIndex / params.headDimension;
  let dimension = outputIndex % params.headDimension;
  let scale = inverseSqrt(f32(params.headDimension));

  var maxScore = -3.4028234663852886e38;
  for (var keyPosition = 0u; keyPosition <= params.position; keyPosition = keyPosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[tokenHiddenIndex(head, scoreDimension)] *
        k[cacheHiddenIndex(keyPosition, head, scoreDimension)];
    }
    maxScore = max(maxScore, score * scale);
  }

  var sum = 0.0;
  var weightedValue = 0.0;
  for (var valuePosition = 0u; valuePosition <= params.position; valuePosition = valuePosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[tokenHiddenIndex(head, scoreDimension)] *
        k[cacheHiddenIndex(valuePosition, head, scoreDimension)];
    }
    let probabilityNumerator = exp(score * scale - maxScore);
    sum = sum + probabilityNumerator;
    weightedValue = weightedValue + probabilityNumerator * v[cacheHiddenIndex(valuePosition, head, dimension)];
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

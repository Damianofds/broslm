import type { Qwen2ModelKvCache } from "./attentionCache";
import { resetQwen2ModelKvCache } from "./attentionCache";
import type { LoadedQwen2Model, TensorView } from "./loader";
import { isFloat32TensorView, type QwenTensorView } from "./quantizedTensor";
import {
  createStaticStorageBuffer,
  createStorageBuffer,
  destroyBuffers,
  encodeComputeShader,
  readFloat32Buffer,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

interface ResidentQwen2GpuContext {
  temps: GPUBuffer[];
  encoder: GPUCommandEncoder;
  dummyBias: GPUBuffer;
}

interface ResidentQwen2GpuLayerCache {
  keys: GPUBuffer;
  values: GPUBuffer;
  length: number;
}

interface ResidentQwen2GpuModelCache {
  runtime: WebGpuRuntime;
  maximumSequenceLength: number;
  hiddenSize: number;
  keyValueHiddenSize: number;
  inputIds: number[];
  layers: ResidentQwen2GpuLayerCache[];
}

const residentQwen2GpuCacheByCpuCache = new WeakMap<
  Qwen2ModelKvCache,
  ResidentQwen2GpuModelCache
>();

export async function qwen2PrefillLogitsResidentGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  resetQwen2ModelKvCache(cache);
  const gpuCache = resetResidentQwen2GpuCache(model, cache, runtime);

  const sequenceLength = inputIds.length;
  const hiddenSize = model.config.hiddenSize;
  const hiddenElements = sequenceLength * hiddenSize;
  const intermediateElements = sequenceLength * model.config.intermediateSize;
  const context = createResidentQwen2GpuContext(runtime);

  const tokenIdsBuffer = tempBuffer(context, runtime, Uint32Array.from(inputIds));
  const hiddenA = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const hiddenB = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const normed = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const q = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const attentionOutput = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const gate = tempBuffer(context, runtime, intermediateElements * Float32Array.BYTES_PER_ELEMENT);
  const up = tempBuffer(context, runtime, intermediateElements * Float32Array.BYTES_PER_ELEMENT);
  const activatedGate = tempBuffer(
    context,
    runtime,
    intermediateElements * Float32Array.BYTES_PER_ELEMENT,
  );
  const gated = tempBuffer(context, runtime, intermediateElements * Float32Array.BYTES_PER_ELEMENT);
  const mlpOutput = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const finalHidden = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const logits = tempBuffer(
    context,
    runtime,
    model.config.vocabularySize * Float32Array.BYTES_PER_ELEMENT,
  );

  encodeQwen2EmbeddingSequence(context, runtime, {
    tokenIds: tokenIdsBuffer,
    embedding: model.weights.tokenEmbedding,
    output: hiddenA,
    sequenceLength,
    embeddingSize: hiddenSize,
  });

  let layerInput = hiddenA;
  let layerOutput = hiddenB;

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = gpuCache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing Qwen2 layer/GPU cache at index ${layerIndex}`);
    }

    encodeRmsNormSequence(context, runtime, {
      input: layerInput,
      weight: tensorBuffer(runtime, layerWeights.inputLayerNorm.weight),
      output: normed,
      sequenceLength,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.attention.qProjWeight,
      bias: layerWeights.attention.qProjBias,
      output: q,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.attention.kProjWeight,
      bias: layerWeights.attention.kProjBias,
      output: layerCache.keys,
      inputSize: hiddenSize,
      outputSize: model.config.keyValueHiddenSize,
      sequenceLength,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.attention.vProjWeight,
      bias: layerWeights.attention.vProjBias,
      output: layerCache.values,
      inputSize: hiddenSize,
      outputSize: model.config.keyValueHiddenSize,
      sequenceLength,
    });
    encodeRoPESequence(context, runtime, {
      values: q,
      sequenceLength,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      headDimension: model.config.headDimension,
      basePosition: 0,
      theta: model.config.ropeTheta,
    });
    encodeRoPESequence(context, runtime, {
      values: layerCache.keys,
      sequenceLength,
      hiddenSize: model.config.keyValueHiddenSize,
      numberOfHeads: model.config.numberOfKeyValueHeads,
      headDimension: model.config.headDimension,
      basePosition: 0,
      theta: model.config.ropeTheta,
    });
    encodeCausalGqaAttention(context, runtime, {
      q,
      k: layerCache.keys,
      v: layerCache.values,
      output: attentionOutput,
      sequenceLength,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      numberOfKeyValueHeads: model.config.numberOfKeyValueHeads,
      headDimension: model.config.headDimension,
      keyValueHiddenSize: model.config.keyValueHiddenSize,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: attentionOutput,
      weight: layerWeights.attention.outProjWeight,
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

    encodeRmsNormSequence(context, runtime, {
      input: attentionOutput,
      weight: tensorBuffer(runtime, layerWeights.postAttentionLayerNorm.weight),
      output: normed,
      sequenceLength,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.mlp.gateProjWeight,
      output: gate,
      inputSize: hiddenSize,
      outputSize: model.config.intermediateSize,
      sequenceLength,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.mlp.upProjWeight,
      output: up,
      inputSize: hiddenSize,
      outputSize: model.config.intermediateSize,
      sequenceLength,
    });
    encodeSilu(context, runtime, {
      input: gate,
      output: activatedGate,
      length: intermediateElements,
    });
    encodeElementwiseMultiply(context, runtime, {
      left: activatedGate,
      right: up,
      output: gated,
      length: intermediateElements,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: gated,
      weight: layerWeights.mlp.downProjWeight,
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

  encodeRmsNormSequence(context, runtime, {
    input: layerInput,
    inputOffset: (sequenceLength - 1) * hiddenSize,
    weight: tensorBuffer(runtime, model.weights.finalNorm.weight),
    output: finalHidden,
    sequenceLength: 1,
    featureSize: hiddenSize,
    epsilon: model.config.rmsNormEpsilon,
  });
  encodeQwen2MatrixMultiplySequence(context, runtime, {
    input: finalHidden,
    weight: model.weights.lmHead,
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
      throw new Error(`missing Qwen2 CPU/GPU cache at layer ${layerIndex}`);
    }
    cpuLayerCache.length = sequenceLength;
    gpuLayerCache.length = sequenceLength;
  }
  cache.inputIds.push(...inputIds);
  gpuCache.inputIds.push(...inputIds);
  destroyBuffers(...context.temps);
  return logitsCpu;
}

export async function qwen2DecodeTokenLogitsResidentGpu(
  model: LoadedQwen2Model,
  tokenId: number,
  position: number,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const gpuCache = requireResidentQwen2GpuCache(model, cache, runtime);
  if (cache.inputIds.length !== position) {
    throw new Error(`cache input length is ${cache.inputIds.length}, expected position ${position}`);
  }

  const hiddenSize = model.config.hiddenSize;
  const keyValueHiddenSize = model.config.keyValueHiddenSize;
  const intermediateSize = model.config.intermediateSize;
  const context = createResidentQwen2GpuContext(runtime);
  const tokenIdsBuffer = tempBuffer(context, runtime, Uint32Array.of(tokenId));
  const hiddenA = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const hiddenB = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const normed = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const q = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const k = tempBuffer(context, runtime, keyValueHiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const v = tempBuffer(context, runtime, keyValueHiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const attentionOutput = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const gate = tempBuffer(context, runtime, intermediateSize * Float32Array.BYTES_PER_ELEMENT);
  const up = tempBuffer(context, runtime, intermediateSize * Float32Array.BYTES_PER_ELEMENT);
  const activatedGate = tempBuffer(context, runtime, intermediateSize * Float32Array.BYTES_PER_ELEMENT);
  const gated = tempBuffer(context, runtime, intermediateSize * Float32Array.BYTES_PER_ELEMENT);
  const mlpOutput = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const finalHidden = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const logits = tempBuffer(
    context,
    runtime,
    model.config.vocabularySize * Float32Array.BYTES_PER_ELEMENT,
  );

  encodeQwen2EmbeddingSequence(context, runtime, {
    tokenIds: tokenIdsBuffer,
    embedding: model.weights.tokenEmbedding,
    output: hiddenA,
    sequenceLength: 1,
    embeddingSize: hiddenSize,
  });

  let layerInput = hiddenA;
  let layerOutput = hiddenB;

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const gpuLayerCache = gpuCache.layers[layerIndex];
    const cpuLayerCache = cache.layers[layerIndex];
    if (!layerWeights || !gpuLayerCache || !cpuLayerCache) {
      throw new Error(`missing Qwen2 layer/cache at index ${layerIndex}`);
    }
    if (gpuLayerCache.length !== position || cpuLayerCache.length !== position) {
      throw new Error(
        `Qwen2 cache length mismatch at layer ${layerIndex}: GPU ${gpuLayerCache.length}, ` +
          `CPU ${cpuLayerCache.length}, expected ${position}`,
      );
    }

    encodeRmsNormSequence(context, runtime, {
      input: layerInput,
      weight: tensorBuffer(runtime, layerWeights.inputLayerNorm.weight),
      output: normed,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.attention.qProjWeight,
      bias: layerWeights.attention.qProjBias,
      output: q,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.attention.kProjWeight,
      bias: layerWeights.attention.kProjBias,
      output: k,
      inputSize: hiddenSize,
      outputSize: keyValueHiddenSize,
      sequenceLength: 1,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.attention.vProjWeight,
      bias: layerWeights.attention.vProjBias,
      output: v,
      inputSize: hiddenSize,
      outputSize: keyValueHiddenSize,
      sequenceLength: 1,
    });
    encodeRoPESequence(context, runtime, {
      values: q,
      sequenceLength: 1,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      headDimension: model.config.headDimension,
      basePosition: position,
      theta: model.config.ropeTheta,
    });
    encodeRoPESequence(context, runtime, {
      values: k,
      sequenceLength: 1,
      hiddenSize: keyValueHiddenSize,
      numberOfHeads: model.config.numberOfKeyValueHeads,
      headDimension: model.config.headDimension,
      basePosition: position,
      theta: model.config.ropeTheta,
    });

    const cacheByteOffset = position * keyValueHiddenSize * Float32Array.BYTES_PER_ELEMENT;
    const tokenByteLength = keyValueHiddenSize * Float32Array.BYTES_PER_ELEMENT;
    context.encoder.copyBufferToBuffer(k, 0, gpuLayerCache.keys, cacheByteOffset, tokenByteLength);
    context.encoder.copyBufferToBuffer(v, 0, gpuLayerCache.values, cacheByteOffset, tokenByteLength);

    encodeCausalGqaAttentionDecode(context, runtime, {
      q,
      k: gpuLayerCache.keys,
      v: gpuLayerCache.values,
      output: attentionOutput,
      position,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      numberOfKeyValueHeads: model.config.numberOfKeyValueHeads,
      headDimension: model.config.headDimension,
      keyValueHiddenSize,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: attentionOutput,
      weight: layerWeights.attention.outProjWeight,
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

    encodeRmsNormSequence(context, runtime, {
      input: attentionOutput,
      weight: tensorBuffer(runtime, layerWeights.postAttentionLayerNorm.weight),
      output: normed,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.mlp.gateProjWeight,
      output: gate,
      inputSize: hiddenSize,
      outputSize: intermediateSize,
      sequenceLength: 1,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.mlp.upProjWeight,
      output: up,
      inputSize: hiddenSize,
      outputSize: intermediateSize,
      sequenceLength: 1,
    });
    encodeSilu(context, runtime, {
      input: gate,
      output: activatedGate,
      length: intermediateSize,
    });
    encodeElementwiseMultiply(context, runtime, {
      left: activatedGate,
      right: up,
      output: gated,
      length: intermediateSize,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: gated,
      weight: layerWeights.mlp.downProjWeight,
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

  encodeRmsNormSequence(context, runtime, {
    input: layerInput,
    weight: tensorBuffer(runtime, model.weights.finalNorm.weight),
    output: finalHidden,
    sequenceLength: 1,
    featureSize: hiddenSize,
    epsilon: model.config.rmsNormEpsilon,
  });
  encodeQwen2MatrixMultiplySequence(context, runtime, {
    input: finalHidden,
    weight: model.weights.lmHead,
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
      throw new Error(`missing Qwen2 CPU/GPU cache at layer ${layerIndex}`);
    }
    cpuLayerCache.length = position + 1;
    gpuLayerCache.length = position + 1;
  }
  gpuCache.inputIds.push(tokenId);
  destroyBuffers(...context.temps);
  return logitsCpu;
}

export function hasQwen2ResidentGpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): boolean {
  const gpuCache = residentQwen2GpuCacheByCpuCache.get(cache);
  return Boolean(gpuCache && residentQwen2GpuCacheMatches(model, cache, runtime, gpuCache));
}

function createResidentQwen2GpuContext(runtime: WebGpuRuntime): ResidentQwen2GpuContext {
  const dummyBias = createStorageBuffer(runtime, Float32Array.of(0));
  return {
    temps: [dummyBias],
    encoder: runtime.device.createCommandEncoder(),
    dummyBias,
  };
}

function tempBuffer(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  dataOrByteLength: Float32Array | Uint32Array | Uint8Array | ArrayBuffer | number,
  usage?: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = createStorageBuffer(runtime, dataOrByteLength, usage);
  context.temps.push(buffer);
  return buffer;
}

function getOrCreateResidentQwen2GpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuModelCache {
  const existing = residentQwen2GpuCacheByCpuCache.get(cache);
  if (existing && residentQwen2GpuCacheShapeMatches(model, cache, runtime, existing)) {
    return existing;
  }
  if (existing) {
    destroyResidentQwen2GpuCache(existing);
  }

  const layerByteLength =
    cache.maximumSequenceLength * model.config.keyValueHiddenSize * Float32Array.BYTES_PER_ELEMENT;
  const gpuCache: ResidentQwen2GpuModelCache = {
    runtime,
    maximumSequenceLength: cache.maximumSequenceLength,
    hiddenSize: model.config.hiddenSize,
    keyValueHiddenSize: model.config.keyValueHiddenSize,
    inputIds: [],
    layers: Array.from({ length: model.config.numberOfLayers }, () => ({
      keys: createStorageBuffer(runtime, layerByteLength),
      values: createStorageBuffer(runtime, layerByteLength),
      length: 0,
    })),
  };
  residentQwen2GpuCacheByCpuCache.set(cache, gpuCache);
  return gpuCache;
}

function resetResidentQwen2GpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuModelCache {
  const gpuCache = getOrCreateResidentQwen2GpuCache(model, cache, runtime);
  gpuCache.inputIds.length = 0;
  for (const layer of gpuCache.layers) {
    layer.length = 0;
  }
  return gpuCache;
}

function requireResidentQwen2GpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuModelCache {
  const gpuCache = residentQwen2GpuCacheByCpuCache.get(cache);
  if (!gpuCache || !residentQwen2GpuCacheMatches(model, cache, runtime, gpuCache)) {
    throw new Error("Qwen2 WebGPU decode requires a resident GPU cache initialized by prefill.");
  }
  return gpuCache;
}

function residentQwen2GpuCacheMatches(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  gpuCache: ResidentQwen2GpuModelCache,
): boolean {
  if (!residentQwen2GpuCacheShapeMatches(model, cache, runtime, gpuCache)) {
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

function residentQwen2GpuCacheShapeMatches(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  gpuCache: ResidentQwen2GpuModelCache,
): boolean {
  return (
    gpuCache.runtime === runtime &&
    gpuCache.maximumSequenceLength === cache.maximumSequenceLength &&
    gpuCache.hiddenSize === model.config.hiddenSize &&
    gpuCache.keyValueHiddenSize === model.config.keyValueHiddenSize &&
    gpuCache.layers.length === model.config.numberOfLayers
  );
}

function destroyResidentQwen2GpuCache(gpuCache: ResidentQwen2GpuModelCache): void {
  for (const layer of gpuCache.layers) {
    destroyBuffers(layer.keys, layer.values);
  }
}

function tensorBuffer(runtime: WebGpuRuntime, tensor: TensorView): GPUBuffer {
  return createStaticStorageBuffer(runtime, tensor.data);
}

function qwenTensorBuffer(runtime: WebGpuRuntime, tensor: QwenTensorView): GPUBuffer {
  return createStaticStorageBuffer(runtime, tensor.data);
}

function biasBuffer(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  bias: TensorView | Float32Array | undefined,
): { buffer: GPUBuffer; hasBias: boolean } {
  if (!bias) {
    return { buffer: context.dummyBias, hasBias: false };
  }
  if (bias instanceof Float32Array) {
    return { buffer: tempBuffer(context, runtime, bias), hasBias: true };
  }
  return { buffer: tensorBuffer(runtime, bias), hasBias: true };
}

function paramsBuffer(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  data: Float32Array | Uint32Array,
): GPUBuffer {
  return tempBuffer(context, runtime, data, webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst);
}

function encodeQwen2EmbeddingSequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    tokenIds: GPUBuffer;
    embedding: QwenTensorView;
    output: GPUBuffer;
    sequenceLength: number;
    embeddingSize: number;
  },
): void {
  const [entryCount, embeddingSize] = requireMatrixShape(options.embedding, "embedding");
  if (embeddingSize !== options.embeddingSize) {
    throw new Error(`embedding size is ${embeddingSize}, expected ${options.embeddingSize}`);
  }

  if (isFloat32TensorView(options.embedding)) {
    const params = paramsBuffer(
      context,
      runtime,
      new Uint32Array([options.sequenceLength, embeddingSize, entryCount, 0]),
    );
    encodeComputeShader(
      runtime,
      context.encoder,
      qwen2F32EmbeddingSequenceShader,
      [
        { binding: 0, resource: { buffer: options.tokenIds } },
        { binding: 1, resource: { buffer: qwenTensorBuffer(runtime, options.embedding) } },
        { binding: 2, resource: { buffer: params } },
        { binding: 3, resource: { buffer: options.output } },
      ],
      [Math.ceil((options.sequenceLength * embeddingSize) / 128)],
    );
    return;
  }

  const rowByteLength = quantizedRowByteLength(options.embedding, embeddingSize);
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.sequenceLength,
      embeddingSize,
      rowByteLength,
      options.embedding.type === "q4_0" ? 4 : 8,
      entryCount,
      0,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2QuantizedEmbeddingSequenceShader,
    [
      { binding: 0, resource: { buffer: options.tokenIds } },
      { binding: 1, resource: { buffer: qwenTensorBuffer(runtime, options.embedding) } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil((options.sequenceLength * embeddingSize) / 128)],
  );
}

function encodeQwen2MatrixMultiplySequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputBaseOffset?: number;
    weight: QwenTensorView;
    bias?: TensorView | Float32Array;
    output: GPUBuffer;
    inputSize: number;
    outputSize: number;
    sequenceLength: number;
  },
): void {
  const [weightOutputSize, weightInputSize] = requireMatrixShape(options.weight, "weight");
  if (weightInputSize !== options.inputSize || weightOutputSize !== options.outputSize) {
    throw new Error(
      `weight shape is [${weightOutputSize}, ${weightInputSize}], expected ` +
        `[${options.outputSize}, ${options.inputSize}]`,
    );
  }
  const resolvedBias = biasBuffer(context, runtime, options.bias);

  if (isFloat32TensorView(options.weight)) {
    const params = paramsBuffer(
      context,
      runtime,
      new Uint32Array([
        options.inputSize,
        options.outputSize,
        options.sequenceLength,
        options.inputBaseOffset ?? 0,
        resolvedBias.hasBias ? 1 : 0,
        0,
        0,
        0,
      ]),
    );
    encodeComputeShader(
      runtime,
      context.encoder,
      qwen2F32MatrixMultiplySequenceShader,
      [
        { binding: 0, resource: { buffer: qwenTensorBuffer(runtime, options.weight) } },
        { binding: 1, resource: { buffer: options.input } },
        { binding: 2, resource: { buffer: resolvedBias.buffer } },
        { binding: 3, resource: { buffer: params } },
        { binding: 4, resource: { buffer: options.output } },
      ],
      [Math.ceil((options.sequenceLength * options.outputSize) / 64)],
    );
    return;
  }

  const rowByteLength = quantizedRowByteLength(options.weight, options.inputSize);
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.inputSize,
      options.outputSize,
      options.sequenceLength,
      options.inputBaseOffset ?? 0,
      rowByteLength,
      options.weight.type === "q4_0" ? 4 : 8,
      resolvedBias.hasBias ? 1 : 0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2QuantizedMatrixMultiplySequenceShader,
    [
      { binding: 0, resource: { buffer: qwenTensorBuffer(runtime, options.weight) } },
      { binding: 1, resource: { buffer: options.input } },
      { binding: 2, resource: { buffer: resolvedBias.buffer } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [Math.ceil((options.sequenceLength * options.outputSize) / 64)],
  );
}

function encodeRmsNormSequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputOffset?: number;
    weight: GPUBuffer;
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
    qwen2RmsNormSequenceShader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: { buffer: options.weight } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.sequenceLength / 64)],
  );
}

function encodeRoPESequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    values: GPUBuffer;
    sequenceLength: number;
    hiddenSize: number;
    numberOfHeads: number;
    headDimension: number;
    basePosition: number;
    theta: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Float32Array([
      options.theta,
      options.sequenceLength,
      options.hiddenSize,
      options.numberOfHeads,
      options.headDimension,
      options.basePosition,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2RoPESequenceShader,
    [
      { binding: 0, resource: { buffer: options.values } },
      { binding: 1, resource: { buffer: params } },
    ],
    [Math.ceil((options.sequenceLength * options.numberOfHeads * (options.headDimension / 2)) / 128)],
  );
}

function encodeCausalGqaAttention(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    q: GPUBuffer;
    k: GPUBuffer;
    v: GPUBuffer;
    output: GPUBuffer;
    sequenceLength: number;
    hiddenSize: number;
    numberOfHeads: number;
    numberOfKeyValueHeads: number;
    headDimension: number;
    keyValueHiddenSize: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.sequenceLength,
      options.hiddenSize,
      options.numberOfHeads,
      options.numberOfKeyValueHeads,
      options.headDimension,
      options.keyValueHiddenSize,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2CausalGqaAttentionShader,
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

function encodeCausalGqaAttentionDecode(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    q: GPUBuffer;
    k: GPUBuffer;
    v: GPUBuffer;
    output: GPUBuffer;
    position: number;
    hiddenSize: number;
    numberOfHeads: number;
    numberOfKeyValueHeads: number;
    headDimension: number;
    keyValueHiddenSize: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.position,
      options.hiddenSize,
      options.numberOfHeads,
      options.numberOfKeyValueHeads,
      options.headDimension,
      options.keyValueHiddenSize,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2CausalGqaAttentionDecodeShader,
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

function encodeSilu(
  context: ResidentQwen2GpuContext,
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
    qwen2SiluShader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function encodeElementwiseMultiply(
  context: ResidentQwen2GpuContext,
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
    qwen2ElementwiseMultiplyShader,
    [
      { binding: 0, resource: { buffer: options.left } },
      { binding: 1, resource: { buffer: options.right } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function encodeResidualAdd(
  context: ResidentQwen2GpuContext,
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
    qwen2ResidualAddShader,
    [
      { binding: 0, resource: { buffer: options.left } },
      { binding: 1, resource: { buffer: options.right } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function requireMatrixShape(tensor: QwenTensorView, name: string): [number, number] {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }
  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows <= 0 || columns <= 0) {
    throw new Error(`${name} dimensions must be positive, got [${rows}, ${columns}]`);
  }
  return [rows, columns];
}

function quantizedRowByteLength(
  tensor: Exclude<QwenTensorView, TensorView>,
  inputSize: number,
): number {
  if (!Number.isInteger(inputSize) || inputSize <= 0 || inputSize % 32 !== 0) {
    throw new Error(`${tensor.type} length must be a positive multiple of 32, got ${inputSize}`);
  }
  return tensor.type === "q4_0" ? (inputSize / 32) * 18 : (inputSize / 32) * 34;
}

const qwen2F32EmbeddingSequenceShader = `
struct Params {
  sequenceLength: u32,
  embeddingSize: u32,
  entryCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(1) var<storage, read> embedding: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let elementCount = params.sequenceLength * params.embeddingSize;
  if (index >= elementCount) {
    return;
  }
  let position = index / params.embeddingSize;
  let dimension = index % params.embeddingSize;
  let tokenId = tokenIds[position];
  output[index] = embedding[tokenId * params.embeddingSize + dimension];
}
`;

const qwen2QuantizedEmbeddingSequenceShader = `
struct Params {
  sequenceLength: u32,
  embeddingSize: u32,
  rowByteLength: u32,
  quantType: u32,
  entryCount: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(1) var<storage, read> words: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn byteAt(byteOffset: u32) -> u32 {
  let word = words[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByteAt(byteOffset: u32) -> f32 {
  let value = byteAt(byteOffset);
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let elementCount = params.sequenceLength * params.embeddingSize;
  if (index >= elementCount) {
    return;
  }

  let position = index / params.embeddingSize;
  let dimension = index % params.embeddingSize;
  let tokenId = tokenIds[position];
  let rowOffset = tokenId * params.rowByteLength;
  let block = dimension / 32u;
  let blockIndex = dimension % 32u;
  let blockOffset = rowOffset + block * select(18u, 34u, params.quantType == 8u);
  let scale = f16ToF32(byteAt(blockOffset), byteAt(blockOffset + 1u));

  if (params.quantType == 4u) {
    let packed = byteAt(blockOffset + 2u + (blockIndex % 16u));
    let quantized = select((packed >> 4u) & 0x0fu, packed & 0x0fu, blockIndex < 16u);
    output[index] = f32(i32(quantized) - 8) * scale;
  } else {
    output[index] = signedByteAt(blockOffset + 2u + blockIndex) * scale;
  }
}
`;

const qwen2F32MatrixMultiplySequenceShader = `
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
  var sum = 0.0;
  if (params.hasBias == 1u) {
    sum = bias[row];
  }
  let weightOffset = row * params.inputSize;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;
  for (var column = 0u; column < params.inputSize; column = column + 1u) {
    sum = sum + weight[weightOffset + column] * input[inputOffset + column];
  }
  output[index] = sum;
}
`;

const qwen2QuantizedMatrixMultiplySequenceShader = `
struct Params {
  inputSize: u32,
  outputSize: u32,
  sequenceLength: u32,
  inputBaseOffset: u32,
  rowByteLength: u32,
  quantType: u32,
  hasBias: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn byteAt(byteOffset: u32) -> u32 {
  let word = weightWords[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByteAt(byteOffset: u32) -> f32 {
  let value = byteAt(byteOffset);
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let outputElements = params.sequenceLength * params.outputSize;
  if (index >= outputElements) {
    return;
  }

  let position = index / params.outputSize;
  let row = index % params.outputSize;
  var sum = 0.0;
  if (params.hasBias == 1u) {
    sum = bias[row];
  }
  var sourceOffset = row * params.rowByteLength;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;
  for (var base = 0u; base < params.inputSize; base = base + 32u) {
    let scale = f16ToF32(byteAt(sourceOffset), byteAt(sourceOffset + 1u));
    sourceOffset = sourceOffset + 2u;

    if (params.quantType == 4u) {
      for (var packedIndex = 0u; packedIndex < 16u; packedIndex = packedIndex + 1u) {
        let packed = byteAt(sourceOffset + packedIndex);
        let low = f32(i32(packed & 0x0fu) - 8) * scale;
        let high = f32(i32((packed >> 4u) & 0x0fu) - 8) * scale;
        sum = sum + low * input[inputOffset + base + packedIndex];
        sum = sum + high * input[inputOffset + base + 16u + packedIndex];
      }
      sourceOffset = sourceOffset + 16u;
    } else {
      for (var element = 0u; element < 32u; element = element + 1u) {
        sum = sum + signedByteAt(sourceOffset + element) * scale * input[inputOffset + base + element];
      }
      sourceOffset = sourceOffset + 32u;
    }
  }
  output[index] = sum;
}
`;

const qwen2RmsNormSequenceShader = `
struct Params {
  epsilon: f32,
  inputOffset: f32,
  featureSize: f32,
  sequenceLength: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

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
  var meanSquare = 0.0;
  for (var index = 0u; index < featureSize; index = index + 1u) {
    let value = input[inputBase + index];
    meanSquare = meanSquare + value * value;
  }
  meanSquare = meanSquare / f32(featureSize);
  let scale = inverseSqrt(meanSquare + params.epsilon);

  for (var index = 0u; index < featureSize; index = index + 1u) {
    output[outputBase + index] = input[inputBase + index] * scale * weight[index];
  }
}
`;

const qwen2RoPESequenceShader = `
struct Params {
  theta: f32,
  sequenceLength: f32,
  hiddenSize: f32,
  numberOfHeads: f32,
  headDimension: f32,
  basePosition: f32,
  _padding0: f32,
  _padding1: f32,
}

@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let pairGlobalIndex = globalId.x;
  let sequenceLength = u32(params.sequenceLength);
  let hiddenSize = u32(params.hiddenSize);
  let numberOfHeads = u32(params.numberOfHeads);
  let headDimension = u32(params.headDimension);
  let halfDimension = headDimension / 2u;
  let pairsPerPosition = numberOfHeads * halfDimension;
  let pairCount = sequenceLength * pairsPerPosition;
  if (pairGlobalIndex >= pairCount) {
    return;
  }

  let position = pairGlobalIndex / pairsPerPosition;
  let headPairIndex = pairGlobalIndex % pairsPerPosition;
  let head = headPairIndex / halfDimension;
  let pairIndex = headPairIndex % halfDimension;
  let base = position * hiddenSize + head * headDimension;
  let firstIndex = base + pairIndex;
  let secondIndex = base + halfDimension + pairIndex;
  let absolutePosition = params.basePosition + f32(position);
  let angle = absolutePosition / pow(params.theta, (2.0 * f32(pairIndex)) / params.headDimension);
  let cosine = cos(angle);
  let sine = sin(angle);
  let firstValue = values[firstIndex];
  let secondValue = values[secondIndex];
  values[firstIndex] = firstValue * cosine - secondValue * sine;
  values[secondIndex] = secondValue * cosine + firstValue * sine;
}
`;

const qwen2CausalGqaAttentionShader = `
struct Params {
  sequenceLength: u32,
  hiddenSize: u32,
  numberOfHeads: u32,
  numberOfKeyValueHeads: u32,
  headDimension: u32,
  keyValueHiddenSize: u32,
  _padding0: u32,
  _padding1: u32,
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
  let outputLength = params.sequenceLength * params.hiddenSize;
  if (outputIndex >= outputLength) {
    return;
  }

  let queryPosition = outputIndex / params.hiddenSize;
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

const qwen2CausalGqaAttentionDecodeShader = `
struct Params {
  position: u32,
  hiddenSize: u32,
  numberOfHeads: u32,
  numberOfKeyValueHeads: u32,
  headDimension: u32,
  keyValueHiddenSize: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn qHiddenIndex(head: u32, dimension: u32) -> u32 {
  return head * params.headDimension + dimension;
}

fn kvHiddenIndex(position: u32, keyValueHead: u32, dimension: u32) -> u32 {
  return position * params.keyValueHiddenSize + keyValueHead * params.headDimension + dimension;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let outputIndex = globalId.x;
  if (outputIndex >= params.hiddenSize) {
    return;
  }

  let head = outputIndex / params.headDimension;
  let dimension = outputIndex % params.headDimension;
  let groupSize = params.numberOfHeads / params.numberOfKeyValueHeads;
  let keyValueHead = head / groupSize;
  let scale = inverseSqrt(f32(params.headDimension));

  var maxScore = -3.4028234663852886e38;
  for (var keyPosition = 0u; keyPosition <= params.position; keyPosition = keyPosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[qHiddenIndex(head, scoreDimension)] *
        k[kvHiddenIndex(keyPosition, keyValueHead, scoreDimension)];
    }
    maxScore = max(maxScore, score * scale);
  }

  var sum = 0.0;
  var weightedValue = 0.0;
  for (var valuePosition = 0u; valuePosition <= params.position; valuePosition = valuePosition + 1u) {
    var score = 0.0;
    for (var scoreDimension = 0u; scoreDimension < params.headDimension; scoreDimension = scoreDimension + 1u) {
      score = score +
        q[qHiddenIndex(head, scoreDimension)] *
        k[kvHiddenIndex(valuePosition, keyValueHead, scoreDimension)];
    }
    let probabilityNumerator = exp(score * scale - maxScore);
    sum = sum + probabilityNumerator;
    weightedValue = weightedValue + probabilityNumerator * v[kvHiddenIndex(valuePosition, keyValueHead, dimension)];
  }

  output[outputIndex] = weightedValue / sum;
}
`;

const qwen2SiluShader = `
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
  output[index] = value / (1.0 + exp(-value));
}
`;

const qwen2ElementwiseMultiplyShader = `
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
  output[index] = left[index] * right[index];
}
`;

const qwen2ResidualAddShader = `
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

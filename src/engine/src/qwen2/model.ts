import {
  allocateQwen2ModelKvCache,
  cachePrefixMatches,
  resetQwen2ModelKvCache,
  type Qwen2ModelKvCache,
} from "./attentionCache";
import type { LoadedQwen2Model, QwenTensorView, TensorView } from "./loader";
import {
  embeddingLookupQwen,
  embeddingLookupQwenGpu,
  matrixVectorMultiplyQwen,
  matrixVectorMultiplyQwenGpu,
} from "./quantizedTensor";
import {
  qwen2TransformerLayer,
  qwen2TransformerLayerGpu,
  qwen2TransformerLayerIncremental,
  qwen2TransformerLayerIncrementalGpu,
  qwen2TransformerLayerPrefill,
  qwen2TransformerLayerPrefillGpu,
} from "./transformerLayer";
import { rmsNorm } from "../primitives/rmsNorm";
import { rmsNormGpu } from "../primitives/rmsNorm";
import type { InferenceBackend, WebGpuRuntime } from "../runtime/webgpu";
import { sampleTokenFromLogits, type SamplingOptions } from "../sampling";

export interface NextQwen2TokenResult {
  tokenId: number;
  logits: Float32Array;
}

export function qwen2ModelForward(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
): Float32Array {
  validateInputIds(model, inputIds);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = embedInputIds(model, inputIds);

  for (const layerWeights of model.weights.layers) {
    hiddenState = qwen2TransformerLayer(hiddenState, sequenceLength, model.config, layerWeights);
  }

  const output = new Float32Array(sequenceLength * hiddenSize);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    rmsNorm(hiddenState, model.weights.finalNorm.weight, output, {
      inputOffset: offset,
      outputOffset: offset,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
  }

  return output;
}

export function qwen2LastTokenLogits(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
): Float32Array {
  const hiddenState = qwen2ModelForward(model, inputIds);
  const logits = new Float32Array(model.config.vocabularySize);
  const inputOffset = (inputIds.length - 1) * model.config.hiddenSize;

  matrixVectorMultiplyQwen(model.weights.lmHead, hiddenState, logits, {
    inputOffset,
  });

  return logits;
}

export function qwen2NextToken(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  options: SamplingOptions = {},
): NextQwen2TokenResult {
  const logits = qwen2LastTokenLogits(model, inputIds);
  return {
    tokenId: sampleTokenFromLogits(logits, options),
    logits,
  };
}

export function qwen2NextTokenWithCache(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  options: SamplingOptions = {},
): NextQwen2TokenResult {
  const logits = qwen2LastTokenLogitsWithCache(model, inputIds, cache);
  return {
    tokenId: sampleTokenFromLogits(logits, options),
    logits,
  };
}

export async function qwen2NextTokenWithCacheBackend(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  options: SamplingOptions & {
    backend?: InferenceBackend;
    runtime?: WebGpuRuntime;
  } = {},
): Promise<NextQwen2TokenResult> {
  if (options.backend === "webgpu" && !options.runtime) {
    throw new Error("WebGPU backend requires a WebGpuRuntime.");
  }
  if (options.backend === "webgpu" && options.runtime) {
    const logits = await qwen2LastTokenLogitsWithCacheGpu(model, inputIds, cache, options.runtime);
    return {
      tokenId: sampleTokenFromLogits(logits, options),
      logits,
    };
  }

  return qwen2NextTokenWithCache(model, inputIds, cache, options);
}

export async function qwen2ModelForwardGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  validateInputIds(model, inputIds);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = await embedInputIdsGpu(model, inputIds, runtime);

  for (const layerWeights of model.weights.layers) {
    hiddenState = await qwen2TransformerLayerGpu(
      runtime,
      hiddenState,
      sequenceLength,
      model.config,
      layerWeights,
    );
  }

  const output = new Float32Array(sequenceLength * hiddenSize);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    const tokenOutput = await rmsNormGpu(runtime, hiddenState, model.weights.finalNorm.weight, {
      inputOffset: offset,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    output.set(tokenOutput, offset);
  }

  return output;
}

export async function qwen2LastTokenLogitsGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const hiddenState = await qwen2ModelForwardGpu(model, inputIds, runtime);
  const inputOffset = (inputIds.length - 1) * model.config.hiddenSize;
  return matrixVectorMultiplyQwenGpu(runtime, model.weights.lmHead, hiddenState, {
    inputOffset,
  });
}

export function qwen2LastTokenLogitsWithCache(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache = allocateQwen2ModelKvCache(model.config),
): Float32Array {
  validateInputIds(model, inputIds);
  validateQwen2ModelKvCache(model, cache);
  if (inputIds.length > cache.maximumSequenceLength) {
    throw new Error(
      `input sequence length ${inputIds.length} exceeds cache maximumSequenceLength ${cache.maximumSequenceLength}`,
    );
  }

  if (!cachePrefixMatches(cache.inputIds, inputIds) || cache.inputIds.length === inputIds.length) {
    return prefillLogits(model, inputIds, cache);
  }

  let logits: Float32Array = new Float32Array(model.config.vocabularySize);
  for (let position = cache.inputIds.length; position < inputIds.length; position += 1) {
    const tokenId = inputIds[position] ?? 0;
    logits = decodeTokenLogits(model, tokenId, position, cache);
    cache.inputIds.push(tokenId);
  }

  return logits;
}

function embedInputIds(model: LoadedQwen2Model, inputIds: readonly number[]): Float32Array {
  const { hiddenSize } = model.config;
  const output = new Float32Array(inputIds.length * hiddenSize);
  requireMatrix(model.weights.tokenEmbedding, "tokenEmbedding", model.config.vocabularySize, hiddenSize);

  for (let position = 0; position < inputIds.length; position += 1) {
    embeddingLookupQwen(
      model.weights.tokenEmbedding,
      inputIds[position] ?? 0,
      output,
      position * hiddenSize,
    );
  }

  return output;
}

async function embedInputIdsGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const { hiddenSize } = model.config;
  const output = new Float32Array(inputIds.length * hiddenSize);
  requireMatrix(model.weights.tokenEmbedding, "tokenEmbedding", model.config.vocabularySize, hiddenSize);

  for (let position = 0; position < inputIds.length; position += 1) {
    const embedded = await embeddingLookupQwenGpu(
      runtime,
      model.weights.tokenEmbedding,
      inputIds[position] ?? 0,
    );
    output.set(embedded, position * hiddenSize);
  }

  return output;
}

function prefillLogits(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
): Float32Array {
  resetQwen2ModelKvCache(cache);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = embedInputIds(model, inputIds);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = qwen2TransformerLayerPrefill(
      hiddenState,
      sequenceLength,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const finalHidden = new Float32Array(hiddenSize);
  const lastTokenOffset = (sequenceLength - 1) * hiddenSize;
  rmsNorm(hiddenState, model.weights.finalNorm.weight, finalHidden, {
    inputOffset: lastTokenOffset,
    featureSize: hiddenSize,
    epsilon: model.config.rmsNormEpsilon,
  });

  const logits = new Float32Array(model.config.vocabularySize);
  matrixVectorMultiplyQwen(model.weights.lmHead, finalHidden, logits);
  cache.inputIds.push(...inputIds);
  return logits;
}

async function qwen2LastTokenLogitsWithCacheGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  validateInputIds(model, inputIds);
  validateQwen2ModelKvCache(model, cache);
  if (inputIds.length > cache.maximumSequenceLength) {
    throw new Error(
      `input sequence length ${inputIds.length} exceeds cache maximumSequenceLength ${cache.maximumSequenceLength}`,
    );
  }

  if (!cachePrefixMatches(cache.inputIds, inputIds) || cache.inputIds.length === inputIds.length) {
    return prefillLogitsGpu(model, inputIds, cache, runtime);
  }

  let logits: Float32Array = new Float32Array(model.config.vocabularySize);
  for (let position = cache.inputIds.length; position < inputIds.length; position += 1) {
    const tokenId = inputIds[position] ?? 0;
    logits = await decodeTokenLogitsGpu(model, tokenId, position, cache, runtime);
    cache.inputIds.push(tokenId);
  }

  return logits;
}

async function prefillLogitsGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  resetQwen2ModelKvCache(cache);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = await embedInputIdsGpu(model, inputIds, runtime);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = await qwen2TransformerLayerPrefillGpu(
      runtime,
      hiddenState,
      sequenceLength,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const lastTokenOffset = (sequenceLength - 1) * hiddenSize;
  const finalHidden = await rmsNormGpu(runtime, hiddenState, model.weights.finalNorm.weight, {
    inputOffset: lastTokenOffset,
    featureSize: hiddenSize,
    epsilon: model.config.rmsNormEpsilon,
  });

  const logits = await matrixVectorMultiplyQwenGpu(runtime, model.weights.lmHead, finalHidden);
  cache.inputIds.push(...inputIds);
  return logits;
}

function decodeTokenLogits(
  model: LoadedQwen2Model,
  tokenId: number,
  position: number,
  cache: Qwen2ModelKvCache,
): Float32Array {
  const { hiddenSize } = model.config;
  let hiddenState = embedToken(model, tokenId);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = qwen2TransformerLayerIncremental(
      hiddenState,
      position,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const finalHidden = new Float32Array(hiddenSize);
  rmsNorm(hiddenState, model.weights.finalNorm.weight, finalHidden, {
    featureSize: hiddenSize,
    epsilon: model.config.rmsNormEpsilon,
  });

  const logits = new Float32Array(model.config.vocabularySize);
  matrixVectorMultiplyQwen(model.weights.lmHead, finalHidden, logits);
  return logits;
}

async function decodeTokenLogitsGpu(
  model: LoadedQwen2Model,
  tokenId: number,
  position: number,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const { hiddenSize } = model.config;
  let hiddenState = await embedTokenGpu(model, tokenId, runtime);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = await qwen2TransformerLayerIncrementalGpu(
      runtime,
      hiddenState,
      position,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const finalHidden = await rmsNormGpu(runtime, hiddenState, model.weights.finalNorm.weight, {
    featureSize: hiddenSize,
    epsilon: model.config.rmsNormEpsilon,
  });

  return matrixVectorMultiplyQwenGpu(runtime, model.weights.lmHead, finalHidden);
}

function embedToken(model: LoadedQwen2Model, tokenId: number): Float32Array {
  const { hiddenSize } = model.config;
  const output = new Float32Array(hiddenSize);
  requireMatrix(model.weights.tokenEmbedding, "tokenEmbedding", model.config.vocabularySize, hiddenSize);
  embeddingLookupQwen(model.weights.tokenEmbedding, tokenId, output);
  return output;
}

async function embedTokenGpu(
  model: LoadedQwen2Model,
  tokenId: number,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const { hiddenSize } = model.config;
  requireMatrix(model.weights.tokenEmbedding, "tokenEmbedding", model.config.vocabularySize, hiddenSize);
  return embeddingLookupQwenGpu(runtime, model.weights.tokenEmbedding, tokenId);
}

function validateQwen2ModelKvCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
): void {
  if (cache.layers.length !== model.config.numberOfLayers) {
    throw new Error(
      `cache has ${cache.layers.length} layers, expected ${model.config.numberOfLayers}`,
    );
  }
  if (cache.maximumSequenceLength > model.config.maximumSequenceLength) {
    throw new Error(
      `cache maximumSequenceLength ${cache.maximumSequenceLength} exceeds model context ` +
        `${model.config.maximumSequenceLength}`,
    );
  }
  if (cache.keyValueHiddenSize !== model.config.keyValueHiddenSize) {
    throw new Error(
      `cache keyValueHiddenSize is ${cache.keyValueHiddenSize}, expected ${model.config.keyValueHiddenSize}`,
    );
  }
}

function validateInputIds(model: LoadedQwen2Model, inputIds: readonly number[]): void {
  if (!Array.isArray(inputIds) && !(inputIds instanceof Uint32Array)) {
    throw new Error("inputIds must be an array of token ids");
  }
  if (inputIds.length < 1) {
    throw new Error("inputIds must contain at least one token id");
  }
  if (inputIds.length > model.config.maximumSequenceLength) {
    throw new Error(
      `input sequence length ${inputIds.length} exceeds maximumSequenceLength ` +
        `${model.config.maximumSequenceLength}`,
    );
  }

  for (let index = 0; index < inputIds.length; index += 1) {
    const tokenId = inputIds[index];
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= model.config.vocabularySize) {
      throw new RangeError(
        `inputIds[${index}] must be an integer in [0, ${model.config.vocabularySize}), ` +
          `got ${String(tokenId)}`,
      );
    }
  }
}

function requireMatrix(
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
      `${name} shape is [${tensor.shape.join(", ")}], expected ` +
        `[${expectedRows}, ${expectedColumns}]`,
    );
  }
}

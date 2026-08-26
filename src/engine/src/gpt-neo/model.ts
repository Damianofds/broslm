import {
  allocateModelKvCache,
  cachePrefixMatches,
  resetModelKvCache,
  type ModelKvCache,
} from "./attentionCache";
import { gptNeoPrefillLogitsResidentGpu } from "./gpuModel";
import type { LoadedModel, TensorView } from "./loader";
import { embeddingLookupGpu } from "../primitives/embeddingLookup";
import { layerNorm } from "../primitives/layerNorm";
import { layerNormGpu } from "../primitives/layerNorm";
import { matrixVectorMultiply, matrixVectorMultiplyGpu } from "../primitives/matrixVectorMultiply";
import { residualAddGpu } from "../primitives/residualAdd";
import type { InferenceBackend, WebGpuRuntime } from "../runtime/webgpu";
import { sampleTokenFromLogits, type SamplingOptions } from "../sampling";
import {
  transformerLayer,
  transformerLayerGpu,
  transformerLayerIncremental,
  transformerLayerIncrementalGpu,
  transformerLayerPrefill,
  transformerLayerPrefillGpu,
} from "./transformerLayer";

export interface NextTokenResult {
  tokenId: number;
  logits: Float32Array;
}

export function modelForward(model: LoadedModel, inputIds: readonly number[]): Float32Array {
  validateInputIds(model, inputIds);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = embedInputIds(model, inputIds);

  for (const layerWeights of model.weights.layers) {
    hiddenState = transformerLayer(hiddenState, sequenceLength, model.config, layerWeights);
  }

  const output = new Float32Array(sequenceLength * hiddenSize);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    layerNorm(
      hiddenState,
      model.weights.finalLayerNorm.weight,
      model.weights.finalLayerNorm.bias,
      output,
      {
        inputOffset: offset,
        outputOffset: offset,
        featureSize: hiddenSize,
        epsilon: model.config.layerNormEpsilon,
      },
    );
  }

  return output;
}

export function lastTokenLogits(model: LoadedModel, inputIds: readonly number[]): Float32Array {
  const hiddenState = modelForward(model, inputIds);
  const logits = new Float32Array(model.config.vocabularySize);
  const inputOffset = (inputIds.length - 1) * model.config.hiddenSize;

  matrixVectorMultiply(model.weights.lmHead, hiddenState, logits, {
    inputOffset,
  });

  return logits;
}

export function nextToken(
  model: LoadedModel,
  inputIds: readonly number[],
  options: SamplingOptions = {},
): NextTokenResult {
  const logits = lastTokenLogits(model, inputIds);
  return {
    tokenId: sampleTokenFromLogits(logits, options),
    logits,
  };
}

export function nextTokenWithCache(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
  options: SamplingOptions = {},
): NextTokenResult {
  const logits = lastTokenLogitsWithCache(model, inputIds, cache);
  return {
    tokenId: sampleTokenFromLogits(logits, options),
    logits,
  };
}

export async function nextTokenWithCacheBackend(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
  options: SamplingOptions & {
    backend?: InferenceBackend;
    runtime?: WebGpuRuntime;
  } = {},
): Promise<NextTokenResult> {
  if (options.backend === "webgpu" && !options.runtime) {
    throw new Error("WebGPU backend requires a WebGpuRuntime.");
  }
  if (options.backend === "webgpu" && options.runtime) {
    const logits = await lastTokenLogitsWithCacheGpu(model, inputIds, cache, options.runtime);
    return {
      tokenId: sampleTokenFromLogits(logits, options),
      logits,
    };
  }

  return nextTokenWithCache(model, inputIds, cache, options);
}

export async function modelForwardGpu(
  model: LoadedModel,
  inputIds: readonly number[],
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  validateInputIds(model, inputIds);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = await embedInputIdsGpu(model, inputIds, runtime);

  for (const layerWeights of model.weights.layers) {
    hiddenState = await transformerLayerGpu(
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
    const tokenOutput = await layerNormGpu(
      runtime,
      hiddenState,
      model.weights.finalLayerNorm.weight,
      model.weights.finalLayerNorm.bias,
      {
        inputOffset: offset,
        featureSize: hiddenSize,
        epsilon: model.config.layerNormEpsilon,
      },
    );
    output.set(tokenOutput, offset);
  }

  return output;
}

export async function lastTokenLogitsGpu(
  model: LoadedModel,
  inputIds: readonly number[],
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const hiddenState = await modelForwardGpu(model, inputIds, runtime);
  const inputOffset = (inputIds.length - 1) * model.config.hiddenSize;
  return matrixVectorMultiplyGpu(runtime, model.weights.lmHead, hiddenState, {
    inputOffset,
  });
}

export function lastTokenLogitsWithCache(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache = allocateModelKvCache(model.config),
): Float32Array {
  validateInputIds(model, inputIds);
  validateModelKvCache(model, cache);

  if (!cachePrefixMatches(cache.inputIds, inputIds) || cache.inputIds.length === inputIds.length) {
    return prefillLogits(model, inputIds, cache);
  }

  let logits: Float32Array = new Float32Array(model.config.vocabularySize);
  for (let position = cache.inputIds.length; position < inputIds.length; position += 1) {
    logits = decodeTokenLogits(model, inputIds[position] ?? 0, position, cache);
    cache.inputIds.push(inputIds[position] ?? 0);
  }

  return logits;
}

function embedInputIds(model: LoadedModel, inputIds: readonly number[]): Float32Array {
  const { hiddenSize } = model.config;
  const output = new Float32Array(inputIds.length * hiddenSize);
  const tokenEmbedding = requireMatrix(
    model.weights.tokenEmbedding,
    "tokenEmbedding",
    model.config.vocabularySize,
    hiddenSize,
  );
  const positionEmbedding = requireMatrix(
    model.weights.positionEmbedding,
    "positionEmbedding",
    model.config.maximumSequenceLength,
    hiddenSize,
  );

  for (let position = 0; position < inputIds.length; position += 1) {
    const tokenId = inputIds[position] ?? 0;
    const outputOffset = position * hiddenSize;
    const tokenOffset = tokenId * hiddenSize;
    const positionOffset = position * hiddenSize;

    for (let dimension = 0; dimension < hiddenSize; dimension += 1) {
      output[outputOffset + dimension] =
        (tokenEmbedding.data[tokenOffset + dimension] ?? 0) +
        (positionEmbedding.data[positionOffset + dimension] ?? 0);
    }
  }

  return output;
}

async function embedInputIdsGpu(
  model: LoadedModel,
  inputIds: readonly number[],
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const { hiddenSize } = model.config;
  const output = new Float32Array(inputIds.length * hiddenSize);
  const tokenEmbedding = requireMatrix(
    model.weights.tokenEmbedding,
    "tokenEmbedding",
    model.config.vocabularySize,
    hiddenSize,
  );
  const positionEmbedding = requireMatrix(
    model.weights.positionEmbedding,
    "positionEmbedding",
    model.config.maximumSequenceLength,
    hiddenSize,
  );

  for (let position = 0; position < inputIds.length; position += 1) {
    const token = await embeddingLookupGpu(runtime, tokenEmbedding, inputIds[position] ?? 0);
    const positionVector = await embeddingLookupGpu(runtime, positionEmbedding, position);
    const embedded = await residualAddGpu(runtime, token, positionVector);
    output.set(embedded, position * hiddenSize);
  }

  return output;
}

function prefillLogits(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
): Float32Array {
  resetModelKvCache(cache);

  const sequenceLength = inputIds.length;
  const { hiddenSize } = model.config;
  let hiddenState = embedInputIds(model, inputIds);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = transformerLayerPrefill(
      hiddenState,
      sequenceLength,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const finalHidden = new Float32Array(hiddenSize);
  const lastTokenOffset = (sequenceLength - 1) * hiddenSize;
  layerNorm(
    hiddenState,
    model.weights.finalLayerNorm.weight,
    model.weights.finalLayerNorm.bias,
    finalHidden,
    {
      inputOffset: lastTokenOffset,
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    },
  );

  const logits = new Float32Array(model.config.vocabularySize);
  matrixVectorMultiply(model.weights.lmHead, finalHidden, logits);
  cache.inputIds.push(...inputIds);
  return logits;
}

async function lastTokenLogitsWithCacheGpu(
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  validateInputIds(model, inputIds);
  validateModelKvCache(model, cache);

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
  model: LoadedModel,
  inputIds: readonly number[],
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  return gptNeoPrefillLogitsResidentGpu(model, inputIds, cache, runtime);
}

function decodeTokenLogits(
  model: LoadedModel,
  tokenId: number,
  position: number,
  cache: ModelKvCache,
): Float32Array {
  const { hiddenSize } = model.config;
  let hiddenState = embedTokenAtPosition(model, tokenId, position);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = transformerLayerIncremental(
      hiddenState,
      position,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const finalHidden = new Float32Array(hiddenSize);
  layerNorm(
    hiddenState,
    model.weights.finalLayerNorm.weight,
    model.weights.finalLayerNorm.bias,
    finalHidden,
    {
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    },
  );

  const logits = new Float32Array(model.config.vocabularySize);
  matrixVectorMultiply(model.weights.lmHead, finalHidden, logits);
  return logits;
}

async function decodeTokenLogitsGpu(
  model: LoadedModel,
  tokenId: number,
  position: number,
  cache: ModelKvCache,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const { hiddenSize } = model.config;
  let hiddenState = await embedTokenAtPositionGpu(model, tokenId, position, runtime);

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = cache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing layer/cache at index ${layerIndex}`);
    }
    hiddenState = await transformerLayerIncrementalGpu(
      runtime,
      hiddenState,
      position,
      model.config,
      layerWeights,
      layerCache,
    );
  }

  const finalHidden = await layerNormGpu(
    runtime,
    hiddenState,
    model.weights.finalLayerNorm.weight,
    model.weights.finalLayerNorm.bias,
    {
      featureSize: hiddenSize,
      epsilon: model.config.layerNormEpsilon,
    },
  );

  return matrixVectorMultiplyGpu(runtime, model.weights.lmHead, finalHidden);
}

function embedTokenAtPosition(
  model: LoadedModel,
  tokenId: number,
  position: number,
): Float32Array {
  const { hiddenSize } = model.config;
  const output = new Float32Array(hiddenSize);
  const tokenEmbedding = requireMatrix(
    model.weights.tokenEmbedding,
    "tokenEmbedding",
    model.config.vocabularySize,
    hiddenSize,
  );
  const positionEmbedding = requireMatrix(
    model.weights.positionEmbedding,
    "positionEmbedding",
    model.config.maximumSequenceLength,
    hiddenSize,
  );
  const tokenOffset = tokenId * hiddenSize;
  const positionOffset = position * hiddenSize;

  for (let dimension = 0; dimension < hiddenSize; dimension += 1) {
    output[dimension] =
      (tokenEmbedding.data[tokenOffset + dimension] ?? 0) +
      (positionEmbedding.data[positionOffset + dimension] ?? 0);
  }

  return output;
}

async function embedTokenAtPositionGpu(
  model: LoadedModel,
  tokenId: number,
  position: number,
  runtime: WebGpuRuntime,
): Promise<Float32Array> {
  const { hiddenSize } = model.config;
  const tokenEmbedding = requireMatrix(
    model.weights.tokenEmbedding,
    "tokenEmbedding",
    model.config.vocabularySize,
    hiddenSize,
  );
  const positionEmbedding = requireMatrix(
    model.weights.positionEmbedding,
    "positionEmbedding",
    model.config.maximumSequenceLength,
    hiddenSize,
  );

  const token = await embeddingLookupGpu(runtime, tokenEmbedding, tokenId);
  const positionVector = await embeddingLookupGpu(runtime, positionEmbedding, position);
  return residualAddGpu(runtime, token, positionVector);
}

function validateModelKvCache(model: LoadedModel, cache: ModelKvCache): void {
  if (cache.layers.length !== model.config.numberOfLayers) {
    throw new Error(
      `cache has ${cache.layers.length} layers, expected ${model.config.numberOfLayers}`,
    );
  }
  if (cache.maximumSequenceLength !== model.config.maximumSequenceLength) {
    throw new Error(
      `cache maximumSequenceLength is ${cache.maximumSequenceLength}, expected ` +
        `${model.config.maximumSequenceLength}`,
    );
  }
  if (cache.hiddenSize !== model.config.hiddenSize) {
    throw new Error(`cache hiddenSize is ${cache.hiddenSize}, expected ${model.config.hiddenSize}`);
  }
}

function validateInputIds(model: LoadedModel, inputIds: readonly number[]): void {
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
    if (
      !Number.isInteger(tokenId) ||
      tokenId < 0 ||
      tokenId >= model.config.vocabularySize
    ) {
      throw new RangeError(
        `inputIds[${index}] must be an integer in [0, ${model.config.vocabularySize}), ` +
          `got ${String(tokenId)}`,
      );
    }
  }
}

function requireMatrix(
  tensor: TensorView,
  name: string,
  expectedRows: number,
  expectedColumns: number,
): TensorView {
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

  if (tensor.data.length !== rows * columns) {
    throw new Error(`${name} data length does not match shape [${rows}, ${columns}]`);
  }

  return tensor;
}

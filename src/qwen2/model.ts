import type { BroslmLogger } from "../logger";
import { sampleTokenFromLogits, type SamplingOptions } from "../sampling";
import type { WebGpuRuntime } from "../runtime/webgpu";
import {
  cachePrefixMatches,
  type Qwen2ModelKvCache,
} from "./attentionCache";
import {
  hasQwen2ResidentGpuCache,
  qwen2DecodeTokenLogitsResidentGpu,
  qwen2PrefillLogitsResidentGpu,
} from "./gpuModel";
import type { LoadedQwen2Model } from "./loader";

export interface NextQwen2TokenResult {
  tokenId: number;
  logits: Float32Array;
}

export async function qwen2NextTokenWithCacheGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  logger: BroslmLogger,
  options: SamplingOptions = {},
): Promise<NextQwen2TokenResult> {
  validateInputIds(model, inputIds);
  validateQwen2ModelKvCache(model, cache);
  if (inputIds.length > cache.maximumSequenceLength) {
    throw new Error(
      `input sequence length ${inputIds.length} exceeds cache maximumSequenceLength ${cache.maximumSequenceLength}`,
    );
  }

  let logits: Float32Array;
  if (
    !cachePrefixMatches(cache.inputIds, inputIds) ||
    cache.inputIds.length === inputIds.length ||
    !hasQwen2ResidentGpuCache(model, cache, runtime)
  ) {
    logits = await qwen2PrefillLogitsResidentGpu(model, inputIds, cache, runtime, logger);
  } else {
    logits = new Float32Array(model.config.vocabularySize);
    for (let position = cache.inputIds.length; position < inputIds.length; position += 1) {
      const tokenId = inputIds[position] ?? 0;
      logits = await qwen2DecodeTokenLogitsResidentGpu(model, tokenId, position, cache, runtime);
      cache.inputIds.push(tokenId);
    }
  }

  return {
    tokenId: sampleTokenFromLogits(logits, options),
    logits,
  };
}

function validateQwen2ModelKvCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
): void {
  if (cache.layers.length !== model.config.numberOfLayers) {
    throw new Error(`cache has ${cache.layers.length} layers, expected ${model.config.numberOfLayers}`);
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

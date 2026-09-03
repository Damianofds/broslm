import type { BroslmLogger } from "../logger";
import { sampleTokenFromCandidates, type SamplingOptions } from "../sampling";
import type { WebGpuRuntime } from "../runtime/webgpu";
import { cachePrefixMatches, type Qwen2ModelKvCache } from "./attentionCache";
import {
  hasQwen2ResidentGpuCache,
  qwen2DecodeTokenLogitsResidentGpu,
  qwen2PrefillLogitsResidentGpu,
  type Qwen2GpuTopKResult,
} from "./gpuModel";
import type { LoadedQwen2Model } from "./loader";
import { qwen2WebGpuSafetyLimits } from "./webgpuSafety";

export interface NextQwen2TokenResult {
  tokenId: number;
  logits: Float32Array;
}

export async function qwen2PrefillNextTokenGpu(
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

  const topK = resolveTopK(options.topK);
  let candidates: Qwen2GpuTopKResult | null = null;
  const reusablePrefix =
    cache.inputIds.length > 0 &&
    cache.inputIds.length < inputIds.length &&
    cachePrefixMatches(cache.inputIds, inputIds) &&
    hasQwen2ResidentGpuCache(model, cache, runtime);

  if (reusablePrefix) {
    for (
      let position = cache.inputIds.length;
      position < inputIds.length;
      position += qwen2WebGpuSafetyLimits.prefillChunkTokens
    ) {
      const chunk = inputIds.slice(
        position,
        Math.min(inputIds.length, position + qwen2WebGpuSafetyLimits.prefillChunkTokens),
      );
      const isFinalChunk = position + chunk.length === inputIds.length;
      candidates = await qwen2PrefillLogitsResidentGpu(
        model,
        chunk,
        cache,
        runtime,
        logger,
        isFinalChunk ? topK : null,
        position,
      );
      logPrefillProgress(logger, position + chunk.length, inputIds.length);
    }
  } else {
    const chunkLength = Math.min(inputIds.length, qwen2WebGpuSafetyLimits.prefillChunkTokens);
    const firstChunk = inputIds.slice(0, chunkLength);
    candidates = await qwen2PrefillLogitsResidentGpu(
      model,
      firstChunk,
      cache,
      runtime,
      logger,
      chunkLength === inputIds.length ? topK : null,
    );
    logPrefillProgress(logger, chunkLength, inputIds.length);

    for (
      let position = chunkLength;
      position < inputIds.length;
      position += qwen2WebGpuSafetyLimits.prefillChunkTokens
    ) {
      const chunk = inputIds.slice(
        position,
        Math.min(inputIds.length, position + qwen2WebGpuSafetyLimits.prefillChunkTokens),
      );
      const isFinalChunk = position + chunk.length === inputIds.length;
      candidates = await qwen2PrefillLogitsResidentGpu(
        model,
        chunk,
        cache,
        runtime,
        logger,
        isFinalChunk ? topK : null,
        position,
      );
      logPrefillProgress(logger, position + chunk.length, inputIds.length);
    }
  }
  return sampleCandidates(requireCandidates(candidates), options);
}

export async function qwen2DecodeNextTokenGpu(
  model: LoadedQwen2Model,
  previousTokenId: number,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  options: SamplingOptions = {},
): Promise<NextQwen2TokenResult> {
  validateQwen2ModelKvCache(model, cache);
  validateTokenId(model, previousTokenId, "previousTokenId");
  const position = cache.inputIds.length;
  if (position >= cache.maximumSequenceLength) {
    throw new Error(`decode position ${position} exceeds the cache capacity`);
  }
  const candidates = await qwen2DecodeTokenLogitsResidentGpu(
    model,
    previousTokenId,
    position,
    cache,
    runtime,
    resolveTopK(options.topK),
  );
  cache.inputIds.push(previousTokenId);
  return sampleCandidates(requireCandidates(candidates), options);
}

export const qwen2NextTokenWithCacheGpu = qwen2PrefillNextTokenGpu;

function sampleCandidates(
  candidates: Qwen2GpuTopKResult,
  options: SamplingOptions,
): NextQwen2TokenResult {
  return {
    tokenId: sampleTokenFromCandidates(candidates.tokenIds, candidates.logits, options),
    logits: candidates.logits,
  };
}

function requireCandidates(candidates: Qwen2GpuTopKResult | null): Qwen2GpuTopKResult {
  if (!candidates) {
    throw new Error("Qwen2 WebGPU prefill completed without producing sampling candidates");
  }
  return candidates;
}

function logPrefillProgress(logger: BroslmLogger, processedTokens: number, totalTokens: number): void {
  if (processedTokens === totalTokens || processedTokens % 256 === 0) {
    logger.debug("webgpu-prefill-progress", { processedTokens, totalTokens });
  }
}

function resolveTopK(value: number | undefined): number {
  return Math.max(1, Math.min(64, Math.round(value ?? 1)));
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
    validateTokenId(model, inputIds[index], `inputIds[${index}]`);
  }
}

function validateTokenId(
  model: LoadedQwen2Model,
  tokenId: number | undefined,
  name: string,
): asserts tokenId is number {
  if (
    typeof tokenId !== "number" ||
    !Number.isInteger(tokenId) ||
    tokenId < 0 ||
    tokenId >= model.config.vocabularySize
  ) {
    throw new RangeError(
      `${name} must be an integer in [0, ${model.config.vocabularySize}), got ${String(tokenId)}`,
    );
  }
}

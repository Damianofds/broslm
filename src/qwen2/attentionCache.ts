import type { Qwen2Config } from "./loader";

export interface Qwen2LayerKvCache {
  length: number;
}

export interface Qwen2ModelKvCache {
  layers: Qwen2LayerKvCache[];
  inputIds: number[];
  maximumSequenceLength: number;
  keyValueHiddenSize: number;
}

export function allocateQwen2ModelKvCache(
  config: Qwen2Config,
  maximumSequenceLength = config.maximumSequenceLength,
): Qwen2ModelKvCache {
  if (!Number.isInteger(maximumSequenceLength) || maximumSequenceLength <= 0) {
    throw new Error(`maximumSequenceLength must be a positive integer, got ${maximumSequenceLength}`);
  }
  if (maximumSequenceLength > config.maximumSequenceLength) {
    throw new Error(
      `maximumSequenceLength ${maximumSequenceLength} exceeds model context ${config.maximumSequenceLength}`,
    );
  }

  return {
    layers: Array.from({ length: config.numberOfLayers }, () => ({
      length: 0,
    })),
    inputIds: [],
    maximumSequenceLength,
    keyValueHiddenSize: config.keyValueHiddenSize,
  };
}

export function resetQwen2ModelKvCache(cache: Qwen2ModelKvCache): void {
  for (const layer of cache.layers) {
    layer.length = 0;
  }
  cache.inputIds.length = 0;
}

export function cachePrefixMatches(
  cacheInputIds: readonly number[],
  inputIds: readonly number[],
): boolean {
  if (cacheInputIds.length > inputIds.length) {
    return false;
  }

  for (let index = 0; index < cacheInputIds.length; index += 1) {
    if (cacheInputIds[index] !== inputIds[index]) {
      return false;
    }
  }

  return true;
}

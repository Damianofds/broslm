import type { ModelConfig } from "./loader";

export interface LayerKvCache {
  keys: Float32Array;
  values: Float32Array;
  length: number;
}

export interface ModelKvCache {
  layers: LayerKvCache[];
  inputIds: number[];
  maximumSequenceLength: number;
  hiddenSize: number;
}

export function allocateModelKvCache(config: ModelConfig): ModelKvCache {
  const layerElements = config.maximumSequenceLength * config.hiddenSize;
  return {
    layers: Array.from({ length: config.numberOfLayers }, () => ({
      keys: new Float32Array(layerElements),
      values: new Float32Array(layerElements),
      length: 0,
    })),
    inputIds: [],
    maximumSequenceLength: config.maximumSequenceLength,
    hiddenSize: config.hiddenSize,
  };
}

export function resetModelKvCache(cache: ModelKvCache): void {
  for (const layer of cache.layers) {
    layer.length = 0;
  }
  cache.inputIds.length = 0;
}

export function cachePrefixMatches(cacheInputIds: readonly number[], inputIds: readonly number[]): boolean {
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

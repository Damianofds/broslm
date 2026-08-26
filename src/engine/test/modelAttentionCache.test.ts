/// <reference types="node" />

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { allocateModelKvCache } from "../src/gpt-neo/attentionCache";
import {
  allocateRuntimeScratch,
  bindModelWeights,
  createTensorViews,
  type LoadedModel,
  type ModelConfig,
  validateConfig,
  validateWeightsBuffer,
  validateWeightsIndex,
  type WeightsIndex,
} from "../src/gpt-neo/loader";
import { lastTokenLogits, nextToken, nextTokenWithCache } from "../src/gpt-neo/model";

const promptIds = [464, 1310, 2576, 1043, 257];
const alternatePromptIds = [13787, 2227, 284, 1382, 257];
const maxSequenceLengthUnderTest = 12;

describe("attention cache model decode", () => {
  let model: LoadedModel;

  beforeAll(() => {
    model = loadTinyStoriesModel(maxSequenceLengthUnderTest);
  });

  it("matches full recompute logits for cached prefill", () => {
    const cache = allocateModelKvCache(model.config);

    const cached = nextTokenWithCache(model, promptIds, cache);
    const fullLogits = lastTokenLogits(model, promptIds);

    expect(cached.tokenId).toBe(nextToken(model, promptIds).tokenId);
    expectLogitsClose(cached.logits, fullLogits);
    expect(cache.inputIds).toEqual(promptIds);
    for (const layer of cache.layers) {
      expect(layer.length).toBe(promptIds.length);
    }
  });

  it("matches full recompute logits after decoding one appended token", () => {
    const cache = allocateModelKvCache(model.config);
    const first = nextTokenWithCache(model, promptIds, cache);
    const extendedInputIds = [...promptIds, first.tokenId];

    const cached = nextTokenWithCache(model, extendedInputIds, cache);
    const fullLogits = lastTokenLogits(model, extendedInputIds);

    expect(cached.tokenId).toBe(nextToken(model, extendedInputIds).tokenId);
    expectLogitsClose(cached.logits, fullLogits);
    expect(cache.inputIds).toEqual(extendedInputIds);
    for (const layer of cache.layers) {
      expect(layer.length).toBe(extendedInputIds.length);
    }
  });

  it("greedily generates the same token sequence as full recompute", () => {
    const cache = allocateModelKvCache(model.config);
    const fullInputIds = [...alternatePromptIds];
    const cachedInputIds = [...alternatePromptIds];
    const fullGenerated: number[] = [];
    const cachedGenerated: number[] = [];

    for (let index = 0; index < 5; index += 1) {
      const full = nextToken(model, fullInputIds);
      const cached = nextTokenWithCache(model, cachedInputIds, cache);

      fullGenerated.push(full.tokenId);
      cachedGenerated.push(cached.tokenId);
      fullInputIds.push(full.tokenId);
      cachedInputIds.push(cached.tokenId);
    }

    expect(cachedGenerated).toEqual(fullGenerated);
  });

  it("resets when the incoming prompt no longer matches the cached prefix", () => {
    const cache = allocateModelKvCache(model.config);
    nextTokenWithCache(model, promptIds, cache);

    const cached = nextTokenWithCache(model, alternatePromptIds, cache);
    const full = nextToken(model, alternatePromptIds);

    expect(cached.tokenId).toBe(full.tokenId);
    expect(cache.inputIds).toEqual(alternatePromptIds);
    for (const layer of cache.layers) {
      expect(layer.length).toBe(alternatePromptIds.length);
    }
  });
});

function expectLogitsClose(actual: Float32Array, expected: Float32Array): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    expect(actual[index] ?? 0).toBeCloseTo(expected[index] ?? 0, 4);
  }
}

function loadTinyStoriesModel(scratchSequenceLength: number): LoadedModel {
  const modelDir = fileURLToPath(
    new URL("../../../models/output_20260726_105535/", import.meta.url),
  );
  const config = readJson<ModelConfig>(`${modelDir}/config.json`);
  const weightsIndex = readJson<WeightsIndex>(`${modelDir}/weights.json`);
  const weightsBuffer = readArrayBuffer(`${modelDir}/weights.bin`);

  validateConfig(config);
  validateWeightsIndex(config, weightsIndex);
  validateWeightsBuffer(weightsBuffer, weightsIndex);

  const tensors = createTensorViews(weightsBuffer, weightsIndex);
  const weights = bindModelWeights(config, tensors);

  return {
    config,
    weightsIndex,
    weightsBuffer,
    tensors,
    weights,
    scratch: allocateRuntimeScratch(config, scratchSequenceLength),
  };
}

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function readArrayBuffer(path: string): ArrayBuffer {
  const bytes = readFileSync(path);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

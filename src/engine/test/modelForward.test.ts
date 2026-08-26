/// <reference types="node" />

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
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
import { nextToken } from "../src/gpt-neo/model";

interface NextTokenFixture {
  name: string;
  prompt: string;
  inputIds: readonly number[];
  expectedNextTokenId: number;
  expectedNextTokenText: string;
}

interface GreedyGenerationFixture {
  name: string;
  prompt: string;
  inputIds: readonly number[];
  expectedCompletionIds: readonly number[];
  expectedCompletionText: string;
}

const tinyStoriesNextTokenFixtures: readonly NextTokenFixture[] = [
  {
    name: "classic story opener",
    prompt: "Once upon a time",
    inputIds: [7454, 2402, 257, 640],
    expectedNextTokenId: 11,
    expectedNextTokenText: ",",
  },
];

const tinyStoriesGenerationFixtures: readonly GreedyGenerationFixture[] = [
  {
    name: "found object",
    prompt: "The little girl found a",
    inputIds: [464, 1310, 2576, 1043, 257],
    expectedCompletionIds: [1263, 11, 2705, 18447, 13],
    expectedCompletionText: " big, soft blanket.",
  },
  {
    name: "building plan",
    prompt: "Tom wanted to build a",
    inputIds: [13787, 2227, 284, 1382, 257],
    expectedCompletionIds: [1263, 10580, 13, 679, 2227, 284],
    expectedCompletionText: " big tower. He wanted to",
  },
  {
    name: "hidden discovery",
    prompt: "Lily opened the wooden door",
    inputIds: [43, 813, 4721, 262, 13510, 3420],
    expectedCompletionIds: [13, 1375, 2497, 257, 1263, 11, 39145, 28774],
    expectedCompletionText: ". She saw a big, fluffy pillow",
  },
  {
    name: "park visit",
    prompt: "Emma and Jack went to the park",
    inputIds: [10161, 2611, 290, 3619, 1816, 284, 262, 3952],
    expectedCompletionIds: [13, 1119, 2497, 257, 1263, 11, 39145, 3290, 13, 198],
    expectedCompletionText: ". They saw a big, fluffy dog.\n",
  },
];

const requiredScratchSequenceLength = Math.max(
  ...tinyStoriesNextTokenFixtures.map((fixture) => fixture.inputIds.length),
  ...tinyStoriesGenerationFixtures.map(
    (fixture) => fixture.inputIds.length + fixture.expectedCompletionIds.length,
  ),
);

describe("TinyStories model forward", () => {
  let model: LoadedModel;

  beforeAll(() => {
    model = loadTinyStoriesModel(requiredScratchSequenceLength);
  });

  it.each(tinyStoriesNextTokenFixtures)(
    "predicts the oracle next token for $name",
    (fixture) => {
      const result = nextToken(model, fixture.inputIds);

      expect(result.tokenId, `${fixture.prompt} -> ${fixture.expectedNextTokenText}`).toBe(
        fixture.expectedNextTokenId,
      );
    },
  );

  it.each(tinyStoriesGenerationFixtures)(
    "keeps $name fixture input and output in the requested token range",
    (fixture) => {
      expect(fixture.inputIds.length).toBeGreaterThanOrEqual(5);
      expect(fixture.inputIds.length).toBeLessThanOrEqual(10);
      expect(fixture.expectedCompletionIds.length).toBeGreaterThanOrEqual(5);
      expect(fixture.expectedCompletionIds.length).toBeLessThanOrEqual(10);
    },
  );

  it.each(tinyStoriesGenerationFixtures)(
    "greedily generates the fixture completion for $name",
    (fixture) => {
      const generatedTokenIds = generateTokenIds(
        model,
        fixture.inputIds,
        fixture.expectedCompletionIds.length,
      );

      expect(
        generatedTokenIds,
        `${fixture.prompt} -> ${fixture.expectedCompletionText}`,
      ).toEqual(fixture.expectedCompletionIds);
    },
  );
});

function generateTokenIds(
  model: LoadedModel,
  inputIds: readonly number[],
  maxNewTokens: number,
): number[] {
  const runningInputIds = [...inputIds];
  const generatedTokenIds: number[] = [];

  for (let tokenIndex = 0; tokenIndex < maxNewTokens; tokenIndex += 1) {
    const result = nextToken(model, runningInputIds);
    const { tokenId } = result;
    generatedTokenIds.push(tokenId);
    runningInputIds.push(tokenId);
  }

  return generatedTokenIds;
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

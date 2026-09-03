import { beforeEach, describe, expect, it, vi } from "vitest";

const gpu = vi.hoisted(() => ({
  prefill: vi.fn(),
  decode: vi.fn(),
}));

vi.mock("../../src/qwen2/gpuModel", () => ({
  hasQwen2ResidentGpuCache: () => true,
  qwen2PrefillLogitsResidentGpu: gpu.prefill,
  qwen2DecodeTokenLogitsResidentGpu: gpu.decode,
}));

import { createBroslmLogger } from "../../src/logger";
import { qwen2PrefillNextTokenGpu } from "../../src/qwen2/model";
import type { LoadedQwen2Model } from "../../src/qwen2/loader";
import type { Qwen2ModelKvCache } from "../../src/qwen2/attentionCache";
import type { WebGpuRuntime } from "../../src/runtime/webgpu";

describe("Qwen2 stateful chunked prefill", () => {
  beforeEach(() => {
    gpu.prefill.mockReset();
    gpu.prefill.mockImplementation(
      async (
        _model: LoadedQwen2Model,
        chunk: readonly number[],
        cache: Qwen2ModelKvCache,
        _runtime: WebGpuRuntime,
        _logger: unknown,
        topK: number | null,
        basePosition = 0,
      ) => {
        if (basePosition === 0) cache.inputIds.length = 0;
        cache.inputIds.push(...chunk);
        return topK === null
          ? null
          : { tokenIds: Uint32Array.of(7), logits: Float32Array.of(1) };
      },
    );
  });

  it("bounds temporary prefill work and samples only after the final chunk", async () => {
    const inputIds = Array.from({ length: 600 }, (_, index) => index % 100);
    const cache = createCache();

    const result = await qwen2PrefillNextTokenGpu(
      createModel(),
      inputIds,
      cache,
      {} as WebGpuRuntime,
      createBroslmLogger("warn"),
      { temperature: 0, topK: 10 },
    );

    expect(result.tokenId).toBe(7);
    expect(gpu.prefill.mock.calls.map((call) => call[1].length)).toEqual([256, 256, 88]);
    expect(gpu.prefill.mock.calls.map((call) => call[6] ?? 0)).toEqual([0, 256, 512]);
    expect(gpu.prefill.mock.calls.map((call) => call[5])).toEqual([null, null, 10]);
    expect(cache.inputIds).toEqual(inputIds);
  });

  it("prefills only an uncached suffix", async () => {
    const inputIds = Array.from({ length: 400 }, (_, index) => index % 100);
    const cache = createCache(inputIds.slice(0, 300));

    await qwen2PrefillNextTokenGpu(
      createModel(),
      inputIds,
      cache,
      {} as WebGpuRuntime,
      createBroslmLogger("warn"),
      { temperature: 0, topK: 1 },
    );

    expect(gpu.prefill).toHaveBeenCalledOnce();
    expect(gpu.prefill.mock.calls[0]?.[1]).toEqual(inputIds.slice(300));
    expect(gpu.prefill.mock.calls[0]?.[6]).toBe(300);
    expect(cache.inputIds).toEqual(inputIds);
  });
});

function createCache(inputIds: number[] = []): Qwen2ModelKvCache {
  return {
    layers: [{ length: inputIds.length }],
    inputIds: [...inputIds],
    maximumSequenceLength: 1_000,
    keyValueHiddenSize: 64,
  };
}

function createModel(): LoadedQwen2Model {
  return {
    config: {
      architecture: "qwen2",
      vocabularySize: 100,
      hiddenSize: 64,
      intermediateSize: 128,
      numberOfLayers: 1,
      numberOfHeads: 1,
      numberOfKeyValueHeads: 1,
      headDimension: 64,
      keyValueHiddenSize: 64,
      maximumSequenceLength: 1_000,
      rmsNormEpsilon: 1e-6,
      ropeTheta: 1_000_000,
      activation: "silu",
      tiedWordEmbeddings: true,
      bosTokenId: null,
      eosTokenId: 99,
      padTokenId: null,
    },
  } as unknown as LoadedQwen2Model;
}

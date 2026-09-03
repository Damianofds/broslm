import { afterEach, describe, expect, it, vi } from "vitest";
import { createBroslmClient, type BroslmClientDependencies } from "../src/client";
import { BroslmError } from "../src/errors";
import type { BroslmEnvironment } from "../src/environment";
import type { LoadedQwen2Model } from "../src/qwen2/loader";
import type { WebGpuRuntime } from "../src/runtime/webgpu";
import type { ByteLevelBpeTokenizer } from "../src/tokenizer";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("broSLM browser client", () => {
  it("loads and generates exclusively through WebGPU", async () => {
    const info = vi.spyOn(console, "info").mockImplementation(() => undefined);
    const tokenIds = [3, 9];
    const dependencies = testDependencies();
    const nextToken = async () => ({
      tokenId: tokenIds.shift() ?? 9,
      logits: new Float32Array([0]),
    });
    dependencies.prefill = vi.fn(nextToken);
    dependencies.decodeToken = vi.fn(nextToken);
    const client = createBroslmClient(testEnvironment(), { logLevel: "info" }, dependencies);

    const summary = await client.loadModel("qwen");
    expect(summary).toMatchObject({
      modelId: "qwen",
      backend: "webgpu",
      layers: 1,
      attentionHeads: 1,
      keyValueHeads: 1,
    });

    const result = await client.generate("hello", { maxTokens: 3, temperature: 0, topK: 1 });
    expect(result).toMatchObject({
      text: "hello",
      tokenIds: [3],
      inputTokenCount: 2,
      finishReason: "eos",
    });
    expect(dependencies.createWebGpuRuntime).toHaveBeenCalledOnce();
    expect(dependencies.prefill).toHaveBeenCalledWith(
      expect.anything(),
      expect.any(Array),
      expect.anything(),
      expect.objectContaining({ backend: "webgpu" }),
      expect.anything(),
      expect.objectContaining({ temperature: 0, topK: 1 }),
    );
    expect(dependencies.decodeToken).toHaveBeenCalledOnce();
    expect(dependencies.decodeToken).toHaveBeenCalledWith(
      expect.anything(),
      3,
      expect.anything(),
      expect.objectContaining({ backend: "webgpu" }),
      expect.objectContaining({ temperature: 0, topK: 1 }),
    );

    const messages = info.mock.calls.map(([message]) => message);
    expect(messages).toContain("[broslm] model-load-started");
    expect(messages).toContain("[broslm] backend-selected");
    expect(messages).toContain("[broslm] model-ready");
    expect(messages).toContain("[broslm] generation-completed");
    expect(messages).not.toContain("[broslm] generation-progress");
  });

  it("defaults to WARN and suppresses lifecycle logs", async () => {
    const debug = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const info = vi.spyOn(console, "info").mockImplementation(() => undefined);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const client = createBroslmClient(testEnvironment(), {}, testDependencies());

    await client.loadModel("qwen");

    expect(debug).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });

  it("logs progress events at DEBUG with the library prefix", async () => {
    const debug = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    const client = createBroslmClient(testEnvironment(), { logLevel: "debug" }, testDependencies());

    await client.loadModel("qwen");
    await client.generate("hello", { maxTokens: 1 });

    expect(debug.mock.calls.map(([message]) => message)).toContain(
      "[broslm] model-download-progress",
    );
    expect(debug.mock.calls.map(([message]) => message)).toContain(
      "[broslm] generation-progress",
    );
  });

  it("reports unavailable WebGPU without falling back", async () => {
    const dependencies = testDependencies();
    dependencies.detectWebGpuSupport = vi.fn(async () => ({
      supported: false,
      apiAvailable: false,
      adapterAvailable: false,
      reason: "WebGPU is not available in this browser.",
    }));
    const client = createBroslmClient(testEnvironment(), {}, dependencies);

    await expect(client.checkModelSupport("qwen")).resolves.toMatchObject({
      backend: "webgpu",
      supported: false,
    });
    await expect(client.loadModel("qwen")).rejects.toMatchObject({
      code: "BACKEND_UNAVAILABLE",
    });
    expect(dependencies.createWebGpuRuntime).not.toHaveBeenCalled();
  });

  it("uses the same structured prompt for token counting and generation", async () => {
    const encode = vi.fn(() => [1, 2]);
    const client = createBroslmClient(testEnvironment(), {}, testDependencies(encode));
    const messages = [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hello\n\nworld" },
    ] as const;
    await client.loadModel("qwen");

    expect(client.countPromptTokens(messages)).toBe(2);
    await client.generate(messages, { maxTokens: 1 });

    expect(encode).toHaveBeenCalledTimes(2);
    expect(encode).toHaveBeenNthCalledWith(
      1,
      "<|im_start|>system\nBe concise.<|im_end|>\n" +
        "<|im_start|>user\nHello\n\nworld<|im_end|>\n" +
        "<|im_start|>assistant\n",
    );
    expect(encode.mock.calls[1]).toEqual(encode.mock.calls[0]);
  });

  it("disposes the WebGPU runtime and rejects subsequent operations", async () => {
    const dependencies = testDependencies();
    const client = createBroslmClient(testEnvironment(), {}, dependencies);
    await client.loadModel("qwen");
    client.dispose();

    expect(client.state).toBe("disposed");
    expect(dependencies.destroyWebGpuRuntime).toHaveBeenCalled();
    expect(() => client.countPromptTokens("hello")).toThrow(BroslmError);
    await expect(client.checkModelSupport("qwen")).rejects.toMatchObject({ code: "DISPOSED" });
  });
});

function testEnvironment(): BroslmEnvironment {
  return {
    fetchImpl: vi.fn<typeof fetch>(),
    getWebGpuTarget: vi.fn(async () => ({ gpu: {} as GPU })),
  };
}

function testDependencies(
  encode: ByteLevelBpeTokenizer["encode"] = () => [1, 2],
): BroslmClientDependencies {
  return {
    loadQwen2Model: vi.fn(async (options) => {
      options.onProgress?.({ stage: "gguf-download-started", message: "start" });
      options.onProgress?.({
        stage: "gguf-download-progress",
        message: "progress",
        source: "network",
        loadedBytes: 8,
        totalBytes: 16,
      });
      options.onProgress?.({ stage: "gguf-downloaded", message: "downloaded", loadedBytes: 16 });
      options.onProgress?.({ stage: "gguf-parsed", message: "parsed" });
      options.onProgress?.({ stage: "weights-bound", message: "bound" });
      return tinyModel();
    }),
    createTokenizer: vi.fn(() => ({
      vocabularySize: 10,
      eosTokenId: 9,
      encode,
      decode: () => "hello",
      createIncrementalDecoder: () => ({
        push: () => "hello",
        finish: () => "",
      }),
    })),
    allocateCache: vi.fn((config, maximumSequenceLength = config.maximumSequenceLength) => ({
      layers: Array.from({ length: config.numberOfLayers }, () => ({ length: 0 })),
      inputIds: [],
      maximumSequenceLength,
      keyValueHiddenSize: config.keyValueHiddenSize,
    })),
    prefill: vi.fn(async () => ({ tokenId: 9, logits: new Float32Array([0]) })),
    decodeToken: vi.fn(async () => ({ tokenId: 9, logits: new Float32Array([0]) })),
    detectWebGpuSupport: vi.fn(async () => ({
      supported: true,
      apiAvailable: true,
      adapterAvailable: true,
      limits: {
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 2_147_483_644,
      },
    })),
    createWebGpuRuntime: vi.fn(async () => ({ backend: "webgpu" }) as WebGpuRuntime),
    destroyWebGpuRuntime: vi.fn(),
  };
}

function tinyModel(): LoadedQwen2Model {
  return {
    config: {
      architecture: "qwen2",
      vocabularySize: 10,
      hiddenSize: 2,
      intermediateSize: 4,
      numberOfLayers: 1,
      numberOfHeads: 1,
      numberOfKeyValueHeads: 1,
      headDimension: 2,
      keyValueHiddenSize: 2,
      maximumSequenceLength: 32,
      rmsNormEpsilon: 1e-6,
      ropeTheta: 10_000,
      activation: "silu",
      tiedWordEmbeddings: true,
      bosTokenId: 0,
      eosTokenId: 9,
      padTokenId: null,
    },
    gguf: {
      version: 3,
      metadata: new Map(),
      tensors: new Map(),
      tensorDataOffset: 0,
      alignment: 32,
    },
    weightsBuffer: new ArrayBuffer(16),
    tensors: new Map(),
    weights: {} as LoadedQwen2Model["weights"],
  };
}

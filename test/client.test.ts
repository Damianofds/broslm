import { describe, expect, it, vi } from "vitest";
import {
  createBroslmClient,
  type BroslmClientDependencies,
} from "../src/client";
import { BroslmError } from "../src/errors";
import type { BroslmEnvironment } from "../src/environment";
import type { BroslmEvent } from "../src/events";
import type { LoadedQwen2Model } from "../src/qwen2/loader";
import type { ByteLevelBpeTokenizer } from "../src/tokenizer";

describe("broSLM client", () => {
  it("loads, emits lifecycle events, and generates through the high-level API", async () => {
    const events: BroslmEvent[] = [];
    const tokenIds = [3, 9];
    const client = createBroslmClient(testEnvironment(), { onEvent: (event) => events.push(event) }, {
      ...testDependencies(),
      nextToken: vi.fn(async () => ({
        tokenId: tokenIds.shift() ?? 9,
        logits: new Float32Array([0]),
      })),
    });

    const summary = await client.loadModel("qwen_cpu_small");
    expect(summary).toMatchObject({
      modelId: "qwen_cpu_small",
      backend: "cpu",
      layers: 1,
      attentionHeads: 1,
      keyValueHeads: 1,
    });
    expect(client.countPromptTokens("hello")).toBe(2);

    const result = await client.generate("hello", { maxTokens: 3, temperature: 0, topK: 1 });
    expect(result).toMatchObject({
      text: "hello",
      tokenIds: [3],
      inputTokenCount: 2,
      finishReason: "eos",
    });
    expect(events.map((event) => event.type)).toEqual([
      "model-load-started",
      "backend-selected",
      "model-download-started",
      "model-downloaded",
      "model-parsed",
      "model-weights-bound",
      "model-ready",
      "generation-started",
      "generation-progress",
      "generation-progress",
      "generation-completed",
    ]);
  });

  it("isolates event listener errors and reports cancellation", async () => {
    const controller = new AbortController();
    const events: BroslmEvent[] = [];
    const client = createBroslmClient(testEnvironment(), {
      onEvent: () => {
        throw new Error("listener failed");
      },
    }, testDependencies());
    client.subscribe((event) => events.push(event));
    await client.loadModel("qwen_cpu_small");

    controller.abort();
    await expect(client.generate("hello", { signal: controller.signal })).rejects.toMatchObject({
      code: "ABORTED",
    });
    expect(events.some((event) => event.type === "operation-cancelled")).toBe(true);
  });

  it("uses the same structured prompt for token counting and generation", async () => {
    const encode = vi.fn(() => [1, 2]);
    const client = createBroslmClient(testEnvironment(), {}, testDependencies(encode));
    const messages = [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hello\n\nworld" },
    ] as const;
    await client.loadModel("qwen_cpu_small");

    expect(client.countPromptTokens(messages)).toBe(2);
    await client.generate(messages, { maxTokens: 1 });
    for await (const _chunk of client.stream(messages, { maxTokens: 1 })) {
      // The EOS-only fixture produces no visible chunks.
    }

    expect(encode).toHaveBeenCalledTimes(3);
    expect(encode).toHaveBeenNthCalledWith(
      1,
      "<|im_start|>system\nBe concise.<|im_end|>\n" +
        "<|im_start|>user\nHello\n\nworld<|im_end|>\n" +
        "<|im_start|>assistant\n",
    );
    expect(encode.mock.calls[1]).toEqual(encode.mock.calls[0]);
    expect(encode.mock.calls[2]).toEqual(encode.mock.calls[0]);
  });

  it("rejects invalid structured prompts before generation", async () => {
    const client = createBroslmClient(testEnvironment(), {}, testDependencies());
    await client.loadModel("qwen_cpu_small");

    await expect(client.generate([], { maxTokens: 1 })).rejects.toMatchObject({
      code: "INVALID_ARGUMENT",
    });
    expect(client.state).toBe("ready");
  });

  it("disposes resources and rejects subsequent operations", async () => {
    const release = vi.fn();
    const client = createBroslmClient(testEnvironment(release), {}, testDependencies());
    await client.loadModel("qwen_cpu_small");
    client.dispose();

    expect(client.state).toBe("disposed");
    expect(release).toHaveBeenCalledOnce();
    expect(() => client.countPromptTokens("hello")).toThrow(BroslmError);
    await expect(client.checkModelSupport("qwen_cpu_small")).rejects.toMatchObject({
      code: "DISPOSED",
    });
  });
});

function testEnvironment(release = vi.fn()): BroslmEnvironment {
  return {
    fetchImpl: vi.fn<typeof fetch>(),
    getWebGpuTarget: vi.fn(async () => undefined),
    release,
  };
}

function testDependencies(
  encode: ByteLevelBpeTokenizer["encode"] = () => [1, 2],
): BroslmClientDependencies {
  return {
    loadQwen2Model: vi.fn(async (options) => {
      options.onProgress?.({ stage: "gguf-download-started", message: "start" });
      options.onProgress?.({
        stage: "gguf-downloaded",
        message: "downloaded",
        loadedBytes: 16,
      });
      options.onProgress?.({ stage: "gguf-parsed", message: "parsed" });
      options.onProgress?.({ stage: "weights-bound", message: "bound" });
      options.onProgress?.({ stage: "ready", message: "ready" });
      return tinyModel();
    }),
    createTokenizer: vi.fn(() => ({
      vocabularySize: 10,
      eosTokenId: 9,
      encode,
      decode: () => "hello",
    })),
    allocateCache: vi.fn((config, maximumSequenceLength = config.maximumSequenceLength) => ({
      layers: [],
      inputIds: [],
      maximumSequenceLength,
      keyValueHiddenSize: config.keyValueHiddenSize,
    })),
    resetCache: vi.fn((cache) => {
      cache.inputIds.length = 0;
    }),
    nextToken: vi.fn(async () => ({ tokenId: 9, logits: new Float32Array([0]) })),
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

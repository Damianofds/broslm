import { describe, expect, it, vi } from "vitest";
import {
  bindQwen2ModelWeights,
  loadQwen2Model,
  qwen2ConfigFromGguf,
  validateQwen2Config,
  validateQwen2TensorSet,
  type QwenTensorView,
  type TensorView,
} from "../../src/qwen2/loader";
import {
  GGML_TYPE_F32,
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q8_0,
  type GgufFile,
  type GgufMetadataValue,
  type GgufTensorInfo,
} from "../../src/qwen2/gguf";

describe("Qwen2 loader metadata binding", () => {
  it("reports a missing GGUF clearly when the server returns HTML", async () => {
    await expect(
      loadQwen2Model({
        baseUrl: "https://example.test/models/qwen/",
        fetchImpl: async () =>
          new Response("<!doctype html><title>missing</title>", {
            headers: {
              "content-type": "text/html; charset=utf-8",
            },
          }),
      }),
    ).rejects.toThrow("it is not a GGUF file");
  });

  it("tries fallback GGUF URLs after the local model URL is missing", async () => {
    const fetchImpl = vi.fn<typeof fetch>(async (input) => {
      const url = String(input);
      if (url === "https://example.test/models/qwen/model.gguf") {
        return new Response("missing", { status: 404 });
      }
      return new Response("<!doctype html><title>remote html</title>");
    });

    await expect(
      loadQwen2Model({
        baseUrl: "https://example.test/models/qwen/",
        ggufFallbackUrls: ["https://huggingface.test/qwen.gguf"],
        fetchImpl,
      }),
    ).rejects.toThrow("https://huggingface.test/qwen.gguf");

    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(String(fetchImpl.mock.calls[0]?.[0])).toBe("https://example.test/models/qwen/model.gguf");
    expect(String(fetchImpl.mock.calls[1]?.[0])).toBe("https://huggingface.test/qwen.gguf");
  });

  it("derives the official Qwen2.5 0.5B architecture shape from GGUF metadata", () => {
    const gguf = ggufFile(
      metadataMap([
        ["general.architecture", "qwen2"],
        ["qwen2.embedding_length", 896],
        ["qwen2.feed_forward_length", 4864],
        ["qwen2.block_count", 24],
        ["qwen2.attention.head_count", 14],
        ["qwen2.attention.head_count_kv", 2],
        ["qwen2.context_length", 32768],
        ["qwen2.attention.layer_norm_rms_epsilon", 1e-6],
        ["qwen2.rope.freq_base", 1000000],
        ["tokenizer.ggml.bos_token_id", 151643],
        ["tokenizer.ggml.eos_token_id", 151645],
      ]),
      new Map([
        ["output.weight", tensorInfo("output.weight", [896, 151936], GGML_TYPE_Q8_0)],
      ]),
    );

    const config = qwen2ConfigFromGguf(gguf);

    expect(config.hiddenSize).toBe(896);
    expect(config.intermediateSize).toBe(4864);
    expect(config.numberOfLayers).toBe(24);
    expect(config.numberOfHeads).toBe(14);
    expect(config.numberOfKeyValueHeads).toBe(2);
    expect(config.headDimension).toBe(64);
    expect(config.keyValueHiddenSize).toBe(128);
    expect(config.vocabularySize).toBe(151936);
    expect(config.ropeTheta).toBe(1000000);
    validateQwen2Config(config);
  });

  it("validates and binds the expected Qwen2 tensor names for each layer", () => {
    const config = qwen2ConfigFromGguf(
      ggufFile(
        metadataMap([
          ["general.architecture", "qwen2"],
          ["qwen2.embedding_length", 2],
          ["qwen2.feed_forward_length", 2],
          ["qwen2.block_count", 1],
          ["qwen2.attention.head_count", 1],
          ["qwen2.attention.head_count_kv", 1],
          ["qwen2.context_length", 8],
          ["qwen2.attention.layer_norm_rms_epsilon", 1e-6],
          ["qwen2.rope.freq_base", 10000],
        ]),
        ggufTensorInfos(),
      ),
    );
    const tensors = tensorViews();

    validateQwen2TensorSet(config, ggufFile(new Map(), ggufTensorInfos()), tensors);
    const weights = bindQwen2ModelWeights(config, tensors);

    expect(weights.tokenEmbedding.name).toBe("token_embd.weight");
    expect(weights.layers).toHaveLength(1);
    expect(weights.layers[0]?.attention.qProjBias.name).toBe("blk.0.attn_q.bias");
    expect(weights.layers[0]?.mlp.downProjWeight.name).toBe("blk.0.ffn_down.weight");
    expect(weights.lmHead.name).toBe("output.weight");
  });
});

function ggufTensorInfos(): Map<string, GgufTensorInfo> {
  return new Map([
    ["token_embd.weight", tensorInfo("token_embd.weight", [2, 3], GGML_TYPE_F32)],
    ["output.weight", tensorInfo("output.weight", [2, 3], GGML_TYPE_F32)],
    ["output_norm.weight", tensorInfo("output_norm.weight", [2], GGML_TYPE_F32)],
    ["blk.0.attn_norm.weight", tensorInfo("blk.0.attn_norm.weight", [2], GGML_TYPE_F32)],
    ["blk.0.attn_q.weight", tensorInfo("blk.0.attn_q.weight", [2, 2], GGML_TYPE_F32)],
    ["blk.0.attn_q.bias", tensorInfo("blk.0.attn_q.bias", [2], GGML_TYPE_F32)],
    ["blk.0.attn_k.weight", tensorInfo("blk.0.attn_k.weight", [2, 2], GGML_TYPE_F32)],
    ["blk.0.attn_k.bias", tensorInfo("blk.0.attn_k.bias", [2], GGML_TYPE_F32)],
    ["blk.0.attn_v.weight", tensorInfo("blk.0.attn_v.weight", [2, 2], GGML_TYPE_F32)],
    ["blk.0.attn_v.bias", tensorInfo("blk.0.attn_v.bias", [2], GGML_TYPE_F32)],
    ["blk.0.attn_output.weight", tensorInfo("blk.0.attn_output.weight", [2, 2], GGML_TYPE_F32)],
    ["blk.0.ffn_norm.weight", tensorInfo("blk.0.ffn_norm.weight", [2], GGML_TYPE_F32)],
    ["blk.0.ffn_gate.weight", tensorInfo("blk.0.ffn_gate.weight", [2, 2], GGML_TYPE_F32)],
    ["blk.0.ffn_up.weight", tensorInfo("blk.0.ffn_up.weight", [2, 2], GGML_TYPE_F32)],
    ["blk.0.ffn_down.weight", tensorInfo("blk.0.ffn_down.weight", [2, 2], GGML_TYPE_F32)],
  ]);
}

function tensorViews(): ReadonlyMap<string, QwenTensorView> {
  const tensors = new Map<string, QwenTensorView>();
  for (const [name, info] of ggufTensorInfos()) {
    tensors.set(name, tensor(name, logicalShape(info.dimensions), new Float32Array(product(info.dimensions))));
  }
  return tensors;
}

function ggufFile(metadata: Map<string, GgufMetadataValue>, tensors: Map<string, GgufTensorInfo>): GgufFile {
  return {
    version: 3,
    metadata,
    tensors,
    tensorDataOffset: 0,
    alignment: 32,
  };
}

function metadataMap(entries: Array<[string, GgufMetadataValue]>): Map<string, GgufMetadataValue> {
  return new Map(entries);
}

function tensorInfo(name: string, dimensions: number[], type: number): GgufTensorInfo {
  return {
    name,
    dimensions,
    type,
    offset: 0,
    byteOffset: 0,
    byteLength: 0,
  };
}

function tensor(name: string, shape: number[], data: Float32Array): TensorView {
  return {
    name,
    shape,
    byteOffset: 0,
    byteLength: data.byteLength,
    data,
  };
}

function logicalShape(dimensions: readonly number[]): number[] {
  if (dimensions.length === 2) {
    return [dimensions[1] ?? 0, dimensions[0] ?? 0];
  }
  return [...dimensions];
}

function product(values: readonly number[]): number {
  return values.reduce((total, value) => total * value, 1);
}

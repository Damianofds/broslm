import { describe, expect, it } from "vitest";
import { allocateQwen2ModelKvCache } from "../../src/qwen2/attentionCache";
import type {
  LoadedQwen2Model,
  Qwen2Config,
  Qwen2TransformerLayerWeights,
  TensorView,
} from "../../src/qwen2/loader";
import {
  qwen2LastTokenLogits,
  qwen2LastTokenLogitsWithCache,
  qwen2NextToken,
} from "../../src/qwen2/model";

describe("Qwen2 model forward", () => {
  it("computes logits from token embeddings, final RMSNorm, and lm head", () => {
    const model = tinyModel();
    const logits = qwen2LastTokenLogits(model, [1]);
    const scale = 1 / Math.sqrt((3 * 3 + 4 * 4) / 2 + model.config.rmsNormEpsilon);

    expect(logits[0]).toBeCloseTo(3 * scale, 6);
    expect(logits[1]).toBeCloseTo(4 * scale, 6);
    expect(logits[2]).toBeCloseTo(7 * scale, 6);
    expect(qwen2NextToken(model, [1]).tokenId).toBe(2);
  });

  it("keeps cached decoding equivalent to uncached forward calls", () => {
    const model = tinyModel();
    const cache = allocateQwen2ModelKvCache(model.config, 4);

    const prefillLogits = qwen2LastTokenLogitsWithCache(model, [0, 1], cache);
    const directPrefillLogits = qwen2LastTokenLogits(model, [0, 1]);
    const incrementalLogits = qwen2LastTokenLogitsWithCache(model, [0, 1, 2], cache);
    const directIncrementalLogits = qwen2LastTokenLogits(model, [0, 1, 2]);

    expectArraysClose(prefillLogits, directPrefillLogits);
    expectArraysClose(incrementalLogits, directIncrementalLogits);
    expect(cache.inputIds).toEqual([0, 1, 2]);
    expect(cache.layers[0]?.length).toBe(3);
  });
});

function tinyModel(): LoadedQwen2Model {
  const config: Qwen2Config = {
    architecture: "qwen2",
    vocabularySize: 3,
    hiddenSize: 2,
    intermediateSize: 2,
    numberOfLayers: 1,
    numberOfHeads: 1,
    numberOfKeyValueHeads: 1,
    headDimension: 2,
    keyValueHiddenSize: 2,
    maximumSequenceLength: 8,
    rmsNormEpsilon: 1e-6,
    ropeTheta: 10000,
    activation: "silu",
    tiedWordEmbeddings: true,
    bosTokenId: 0,
    eosTokenId: 2,
    padTokenId: null,
  };
  const weights = {
    tokenEmbedding: tensor("token_embd.weight", [3, 2], new Float32Array([1, 0, 3, 4, -1, 2])),
    layers: [zeroLayerWeights()],
    finalNorm: {
      weight: tensor("output_norm.weight", [2], filled(2, 1)),
    },
    lmHead: tensor("output.weight", [3, 2], new Float32Array([1, 0, 0, 1, 1, 1])),
  };

  return {
    config,
    gguf: {
      version: 3,
      metadata: new Map(),
      tensors: new Map(),
      tensorDataOffset: 0,
      alignment: 32,
    },
    weightsBuffer: new ArrayBuffer(0),
    tensors: new Map(),
    weights,
  };
}

function zeroLayerWeights(): Qwen2TransformerLayerWeights {
  return {
    index: 0,
    inputLayerNorm: {
      weight: tensor("attn_norm.weight", [2], filled(2, 1)),
    },
    attention: {
      qProjWeight: tensor("q", [2, 2], new Float32Array(4)),
      qProjBias: tensor("q_bias", [2], new Float32Array(2)),
      kProjWeight: tensor("k", [2, 2], new Float32Array(4)),
      kProjBias: tensor("k_bias", [2], new Float32Array(2)),
      vProjWeight: tensor("v", [2, 2], new Float32Array(4)),
      vProjBias: tensor("v_bias", [2], new Float32Array(2)),
      outProjWeight: tensor("out", [2, 2], new Float32Array(4)),
    },
    postAttentionLayerNorm: {
      weight: tensor("ffn_norm.weight", [2], filled(2, 1)),
    },
    mlp: {
      gateProjWeight: tensor("gate", [2, 2], new Float32Array(4)),
      upProjWeight: tensor("up", [2, 2], new Float32Array(4)),
      downProjWeight: tensor("down", [2, 2], new Float32Array(4)),
    },
  };
}

function filled(length: number, value: number): Float32Array {
  const data = new Float32Array(length);
  data.fill(value);
  return data;
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

function expectArraysClose(actual: Float32Array, expected: Float32Array): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? 0, 6);
  }
}

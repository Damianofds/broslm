import { describe, expect, it } from "vitest";
import type { Qwen2TransformerLayerWeights, TensorView } from "../../src/qwen2/loader";
import {
  qwen2TransformerLayer,
  qwen2TransformerLayerIncremental,
  qwen2TransformerLayerPrefill,
  type Qwen2TransformerLayerConfig,
} from "../../src/qwen2/transformerLayer";

describe("qwen2TransformerLayer", () => {
  it("preserves the residual stream when attention and MLP projections are zero", () => {
    const input = new Float32Array([1, 3, 5, 9]);

    const output = qwen2TransformerLayer(input, 2, config(), zeroLayerWeights());

    expect(Array.from(output)).toEqual(Array.from(input));
  });

  it("matches prefill and incremental layer execution", () => {
    const input = new Float32Array([1, 3, 5, 9]);
    const weights = zeroLayerWeights();
    const fullCache = layerCache(2);
    const prefilled = qwen2TransformerLayerPrefill(input, 2, config(), weights, fullCache);
    const incremental = new Float32Array(input.length);
    const incrementalCache = layerCache(2);

    for (let position = 0; position < 2; position += 1) {
      incremental.set(
        qwen2TransformerLayerIncremental(
          input.subarray(position * 2, position * 2 + 2),
          position,
          config(),
          weights,
          incrementalCache,
        ),
        position * 2,
      );
    }

    expect(Array.from(prefilled)).toEqual(Array.from(incremental));
    expect(fullCache.length).toBe(2);
    expect(incrementalCache.length).toBe(2);
  });

  it("rejects mismatched sequence lengths clearly", () => {
    expect(() => qwen2TransformerLayer(new Float32Array([1, 2, 3]), 2, config(), zeroLayerWeights()))
      .toThrow("input length is 3, expected 4");
  });
});

function config(): Qwen2TransformerLayerConfig {
  return {
    hiddenSize: 2,
    intermediateSize: 2,
    numberOfHeads: 1,
    numberOfKeyValueHeads: 1,
    headDimension: 2,
    keyValueHiddenSize: 2,
    rmsNormEpsilon: 1e-6,
    ropeTheta: 10000,
  };
}

function zeroLayerWeights(): Qwen2TransformerLayerWeights {
  return {
    index: 0,
    inputLayerNorm: normWeights(2),
    attention: {
      qProjWeight: tensor("q", [2, 2], new Float32Array(4)),
      qProjBias: tensor("q_bias", [2], new Float32Array(2)),
      kProjWeight: tensor("k", [2, 2], new Float32Array(4)),
      kProjBias: tensor("k_bias", [2], new Float32Array(2)),
      vProjWeight: tensor("v", [2, 2], new Float32Array(4)),
      vProjBias: tensor("v_bias", [2], new Float32Array(2)),
      outProjWeight: tensor("out", [2, 2], new Float32Array(4)),
    },
    postAttentionLayerNorm: normWeights(2),
    mlp: {
      gateProjWeight: tensor("gate", [2, 2], new Float32Array(4)),
      upProjWeight: tensor("up", [2, 2], new Float32Array(4)),
      downProjWeight: tensor("down", [2, 2], new Float32Array(4)),
    },
  };
}

function layerCache(sequenceLength: number) {
  return {
    keys: new Float32Array(sequenceLength * config().keyValueHiddenSize),
    values: new Float32Array(sequenceLength * config().keyValueHiddenSize),
    length: 0,
  };
}

function normWeights(hiddenSize: number) {
  return {
    weight: tensor("norm", [hiddenSize], filled(hiddenSize, 1)),
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

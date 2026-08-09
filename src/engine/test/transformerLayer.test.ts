import { describe, expect, it } from "vitest";
import type { TensorView, TransformerLayerWeights } from "../src/loader";
import {
  type TransformerLayerConfig,
  transformerLayer,
} from "../src/transformerLayer";

describe("transformerLayer", () => {
  it("composes ln_1, attention, and the first residual add", () => {
    const output = transformerLayer(
      new Float32Array([1, 3]),
      1,
      config(),
      layerWeights({
        attentionValueWeight: identityMatrix(2),
        attentionOutputWeight: identityMatrix(2),
        mlpUpWeight: zeroMatrix(2, 2),
        mlpDownWeight: zeroMatrix(2, 2),
      }),
    );

    expect(output[0]).toBeCloseTo(0.00000499996, 6);
    expect(output[1]).toBeCloseTo(3.999995, 6);
  });

  it("composes ln_2, per-token MLP, and the second residual add", () => {
    const output = transformerLayer(
      new Float32Array([1, 3, 5, 9]),
      2,
      config(),
      layerWeights({
        attentionValueWeight: zeroMatrix(2, 2),
        attentionOutputWeight: zeroMatrix(2, 2),
        mlpUpWeight: identityMatrix(2),
        mlpDownWeight: identityMatrix(2),
      }),
    );

    expect(output[0]).toBeCloseTo(0.84119158, 6);
    expect(output[1]).toBeCloseTo(3.84118658, 6);
    expect(output[2]).toBeCloseTo(4.84119189, 6);
    expect(output[3]).toBeCloseTo(9.84119064, 6);
  });

  it("returns [sequenceLength, hiddenSize] output", () => {
    const output = transformerLayer(
      new Float32Array([1, 3, 5, 9]),
      2,
      config(),
      layerWeights({
        attentionValueWeight: zeroMatrix(2, 2),
        attentionOutputWeight: zeroMatrix(2, 2),
        mlpUpWeight: zeroMatrix(2, 2),
        mlpDownWeight: zeroMatrix(2, 2),
      }),
    );

    expect(output.length).toBe(2 * 2);
  });

  it("rejects invalid sequence and attention dimensions clearly", () => {
    expect(() =>
      transformerLayer(
        new Float32Array([1, 2, 3]),
        2,
        config(),
        layerWeights({
          attentionValueWeight: zeroMatrix(2, 2),
          attentionOutputWeight: zeroMatrix(2, 2),
          mlpUpWeight: zeroMatrix(2, 2),
          mlpDownWeight: zeroMatrix(2, 2),
        }),
      ),
    ).toThrow("input length is 3, expected 4");

    expect(() =>
      transformerLayer(
        new Float32Array(5),
        1,
        {
          hiddenSize: 5,
          intermediateSize: 2,
          numberOfHeads: 2,
          headDimension: 2,
          layerNormEpsilon: 1e-5,
        },
        layerWeights({
          hiddenSize: 5,
          intermediateSize: 2,
          attentionValueWeight: zeroMatrix(5, 5),
          attentionOutputWeight: zeroMatrix(5, 5),
          mlpUpWeight: zeroMatrix(2, 5),
          mlpDownWeight: zeroMatrix(5, 2),
        }),
      ),
    ).toThrow("hiddenSize 5 must be divisible by numberOfHeads 2");
  });

  it("does not mutate its input", () => {
    const input = new Float32Array([1, 3, 5, 9]);
    const before = Array.from(input);

    transformerLayer(
      input,
      2,
      config(),
      layerWeights({
        attentionValueWeight: zeroMatrix(2, 2),
        attentionOutputWeight: zeroMatrix(2, 2),
        mlpUpWeight: identityMatrix(2),
        mlpDownWeight: identityMatrix(2),
      }),
    );

    expect(Array.from(input)).toEqual(before);
  });
});

function config(): TransformerLayerConfig {
  return {
    hiddenSize: 2,
    intermediateSize: 2,
    numberOfHeads: 1,
    headDimension: 2,
    layerNormEpsilon: 1e-5,
  };
}

function layerWeights(options: {
  hiddenSize?: number;
  intermediateSize?: number;
  attentionValueWeight: Float32Array;
  attentionOutputWeight: Float32Array;
  mlpUpWeight: Float32Array;
  mlpDownWeight: Float32Array;
}): TransformerLayerWeights {
  const hiddenSize = options.hiddenSize ?? 2;
  const intermediateSize = options.intermediateSize ?? 2;

  return {
    index: 0,
    ln1: layerNormWeights(hiddenSize),
    attention: {
      kind: "global",
      qProjWeight: tensor("q_proj.weight", [hiddenSize, hiddenSize], zeroMatrix(hiddenSize, hiddenSize)),
      kProjWeight: tensor("k_proj.weight", [hiddenSize, hiddenSize], zeroMatrix(hiddenSize, hiddenSize)),
      vProjWeight: tensor("v_proj.weight", [hiddenSize, hiddenSize], options.attentionValueWeight),
      outProjWeight: tensor("out_proj.weight", [hiddenSize, hiddenSize], options.attentionOutputWeight),
      outProjBias: tensor("out_proj.bias", [hiddenSize], new Float32Array(hiddenSize)),
    },
    ln2: layerNormWeights(hiddenSize),
    mlp: {
      cFcWeight: tensor("c_fc.weight", [intermediateSize, hiddenSize], options.mlpUpWeight),
      cFcBias: tensor("c_fc.bias", [intermediateSize], new Float32Array(intermediateSize)),
      cProjWeight: tensor("c_proj.weight", [hiddenSize, intermediateSize], options.mlpDownWeight),
      cProjBias: tensor("c_proj.bias", [hiddenSize], new Float32Array(hiddenSize)),
    },
  };
}

function layerNormWeights(hiddenSize: number) {
  return {
    weight: tensor("ln.weight", [hiddenSize], filled(hiddenSize, 1)),
    bias: tensor("ln.bias", [hiddenSize], new Float32Array(hiddenSize)),
  };
}

function identityMatrix(size: number): Float32Array {
  const data = new Float32Array(size * size);
  for (let index = 0; index < size; index += 1) {
    data[index * size + index] = 1;
  }
  return data;
}

function zeroMatrix(rows: number, columns: number): Float32Array {
  return new Float32Array(rows * columns);
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

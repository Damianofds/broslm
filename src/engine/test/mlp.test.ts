import { describe, expect, it } from "vitest";
import type { TensorView } from "../src/gpt-neo/loader";
import {
  type MlpConfig,
  type MlpWeights,
  gptNeoMlpWeights,
  mlp,
} from "../src/gpt-neo/mlp";

describe("mlp", () => {
  it("returns one [hiddenSize] output vector", () => {
    const output = mlp(
      new Float32Array([1, 2]),
      mlpWeights({
        upWeight: new Float32Array(4 * 2),
        downWeight: new Float32Array(2 * 4),
      }),
      { hiddenSize: 2, intermediateSize: 4 },
    );

    expect(output.length).toBe(2);
  });

  it("matches a deterministic tiny MLP with biases", () => {
    const output = mlp(
      new Float32Array([1, 2]),
      mlpWeights({
        // Produces pre-activation intermediate values [-1, 0, 1, 2].
        upWeight: new Float32Array([-1, 0, 0, 1, 1, 0, 0, 1]),
        upBias: new Float32Array([0, -2, 0, 0]),
        downWeight: new Float32Array([1, 1, 0, 0, 0, 0, 1, 1]),
        downBias: new Float32Array([0.5, -0.25]),
      }),
      { hiddenSize: 2, intermediateSize: 4 },
    );

    expect(output[0]).toBeCloseTo(0.34119199, 6);
    expect(output[1]).toBeCloseTo(2.54578972, 6);
  });

  it("returns zero output for zero weights and zero biases", () => {
    const output = mlp(
      new Float32Array([5, -3]),
      mlpWeights({
        upWeight: new Float32Array(4 * 2),
        upBias: new Float32Array(4),
        downWeight: new Float32Array(2 * 4),
        downBias: new Float32Array(2),
      }),
      { hiddenSize: 2, intermediateSize: 4 },
    );

    expect(Array.from(output)).toEqual([0, 0]);
  });

  it("rejects invalid input and weight shapes with clear errors", () => {
    const config: MlpConfig = { hiddenSize: 2, intermediateSize: 4 };

    expect(() => mlp(new Float32Array([1]), validWeights(), config)).toThrow(
      "Invalid MLP input size: expected 2 values, got 1",
    );
    expect(() =>
      mlp(
        new Float32Array([1, 2]),
        {
          ...validWeights(),
          upWeight: tensor("upWeight", [3, 2], new Float32Array(6)),
        },
        config,
      ),
    ).toThrow("Invalid MLP upWeight shape: expected [4, 2], got [3, 2]");
    expect(() =>
      mlp(
        new Float32Array([1, 2]),
        {
          ...validWeights(),
          upWeight: tensor("upWeight", [4, 2], new Float32Array(7)),
        },
        config,
      ),
    ).toThrow("Invalid MLP upWeight size: expected 8 values, got 7");
    expect(() =>
      mlp(
        new Float32Array([1, 2]),
        {
          ...validWeights(),
          upBias: new Float32Array(3),
        },
        config,
      ),
    ).toThrow("Invalid MLP upBias size: expected 4 values, got 3");
    expect(() =>
      mlp(
        new Float32Array([1, 2]),
        {
          ...validWeights(),
          downWeight: tensor("downWeight", [2, 3], new Float32Array(6)),
        },
        config,
      ),
    ).toThrow("Invalid MLP downWeight shape: expected [2, 4], got [2, 3]");
    expect(() =>
      mlp(
        new Float32Array([1, 2]),
        {
          ...validWeights(),
          downWeight: tensor("downWeight", [2, 4], new Float32Array(7)),
        },
        config,
      ),
    ).toThrow("Invalid MLP downWeight size: expected 8 values, got 7");
    expect(() =>
      mlp(
        new Float32Array([1, 2]),
        {
          ...validWeights(),
          downBias: new Float32Array(1),
        },
        config,
      ),
    ).toThrow("Invalid MLP downBias size: expected 2 values, got 1");
  });

  it("does not mutate input or weights", () => {
    const input = new Float32Array([1, 2]);
    const weights = mlpWeights({
      upWeight: new Float32Array([-1, 0, 0, 1, 1, 0, 0, 1]),
      upBias: new Float32Array([0, -2, 0, 0]),
      downWeight: new Float32Array([1, 1, 0, 0, 0, 0, 1, 1]),
      downBias: new Float32Array([0.5, -0.25]),
    });
    const inputBefore = Array.from(input);
    const upWeightBefore = Array.from(weights.upWeight.data);
    const upBiasBefore = Array.from((weights.upBias as TensorView).data);
    const downWeightBefore = Array.from(weights.downWeight.data);
    const downBiasBefore = Array.from((weights.downBias as TensorView).data);

    mlp(input, weights, { hiddenSize: 2, intermediateSize: 4 });

    expect(Array.from(input)).toEqual(inputBefore);
    expect(Array.from(weights.upWeight.data)).toEqual(upWeightBefore);
    expect(Array.from((weights.upBias as TensorView).data)).toEqual(upBiasBefore);
    expect(Array.from(weights.downWeight.data)).toEqual(downWeightBefore);
    expect(Array.from((weights.downBias as TensorView).data)).toEqual(downBiasBefore);
  });

  it("adapts GPT-Neo c_fc and c_proj weights", () => {
    const bound = {
      cFcWeight: tensor("c_fc.weight", [4, 2], new Float32Array(8)),
      cFcBias: tensor("c_fc.bias", [4], new Float32Array(4)),
      cProjWeight: tensor("c_proj.weight", [2, 4], new Float32Array(8)),
      cProjBias: tensor("c_proj.bias", [2], new Float32Array(2)),
    };

    expect(gptNeoMlpWeights(bound)).toEqual({
      upWeight: bound.cFcWeight,
      upBias: bound.cFcBias,
      downWeight: bound.cProjWeight,
      downBias: bound.cProjBias,
    });
  });
});

function validWeights(): MlpWeights {
  return mlpWeights({
    upWeight: new Float32Array(4 * 2),
    upBias: new Float32Array(4),
    downWeight: new Float32Array(2 * 4),
    downBias: new Float32Array(2),
  });
}

function mlpWeights(weights: {
  upWeight: Float32Array;
  upBias?: Float32Array;
  downWeight: Float32Array;
  downBias?: Float32Array;
}): MlpWeights {
  return {
    upWeight: tensor("upWeight", [4, 2], weights.upWeight),
    upBias: weights.upBias ? tensor("upBias", [4], weights.upBias) : undefined,
    downWeight: tensor("downWeight", [2, 4], weights.downWeight),
    downBias: weights.downBias ? tensor("downBias", [2], weights.downBias) : undefined,
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

import { describe, expect, it } from "vitest";
import type { Qwen2MlpWeights, TensorView } from "../../src/qwen2/loader";
import { qwen2MlpWithDebug } from "../../src/qwen2/mlp";

describe("qwen2Mlp", () => {
  it("applies SiLU to the gate projection before elementwise multiplying by the up projection", () => {
    const result = qwen2MlpWithDebug(
      new Float32Array([1, 2]),
      {
        gateProjWeight: tensor("gate", [3, 2], new Float32Array([1, 0, 0, 0, -1, 0])),
        upProjWeight: tensor("up", [3, 2], new Float32Array([2, 0, 0, 1.5, 0, 2])),
        downProjWeight: tensor("down", [2, 3], new Float32Array([1, 1, 1, 0, 1, 0])),
      },
      { hiddenSize: 2, intermediateSize: 3 },
    );

    const expectedGate0 = 1 / (1 + Math.exp(-1));
    const expectedGate2 = -1 / (1 + Math.exp(1));
    expect(result.debug.gate[0]).toBe(1);
    expect(result.debug.up[1]).toBe(3);
    expect(result.debug.gated[0]).toBeCloseTo(expectedGate0 * 2, 6);
    expect(result.debug.gated[2]).toBeCloseTo(expectedGate2 * 4, 6);
    expect(result.output[0]).toBeCloseTo(expectedGate0 * 2 + expectedGate2 * 4, 6);
    expect(result.output[1]).toBe(0);
  });

  it("rejects invalid projection shapes with clear errors", () => {
    const weights = validWeights();

    expect(() =>
      qwen2MlpWithDebug(new Float32Array([1]), weights, { hiddenSize: 2, intermediateSize: 3 }),
    ).toThrow("Invalid Qwen2 MLP input size: expected 2 values, got 1");

    expect(() =>
      qwen2MlpWithDebug(
        new Float32Array([1, 2]),
        {
          ...weights,
          gateProjWeight: tensor("gate", [2, 2], new Float32Array(4)),
        },
        { hiddenSize: 2, intermediateSize: 3 },
      ),
    ).toThrow("Invalid Qwen2 MLP gateProjWeight shape: expected [3, 2], got [2, 2]");
  });
});

function validWeights(): Qwen2MlpWeights {
  return {
    gateProjWeight: tensor("gate", [3, 2], new Float32Array(6)),
    upProjWeight: tensor("up", [3, 2], new Float32Array(6)),
    downProjWeight: tensor("down", [2, 3], new Float32Array(6)),
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

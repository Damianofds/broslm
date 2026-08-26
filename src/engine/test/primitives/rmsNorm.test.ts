import { describe, expect, it } from "vitest";
import type { TensorView } from "../../src/tensor";
import { rmsNorm } from "../../src/primitives/rmsNorm";

describe("rmsNorm", () => {
  it("normalizes by root mean square and applies weight", () => {
    const input = new Float32Array([1, 2, 3]);
    const output = new Float32Array(3);

    rmsNorm(input, tensor("weight", [3], new Float32Array([1, 1.5, 2])), output);

    expect(output[0]).toBeCloseTo(0.46291, 6);
    expect(output[1]).toBeCloseTo(1.38873, 6);
    expect(output[2]).toBeCloseTo(2.77746, 6);
  });

  it("respects offsets and explicit feature size", () => {
    const input = new Float32Array([99, 3, 4, 99]);
    const output = new Float32Array([7, 7, 7, 7]);

    rmsNorm(input, tensor("weight", [2], new Float32Array([2, 0.5])), output, {
      inputOffset: 1,
      outputOffset: 2,
      featureSize: 2,
    });

    expect(output[0]).toBe(7);
    expect(output[1]).toBe(7);
    expect(output[2]).toBeCloseTo(1.697056, 6);
    expect(output[3]).toBeCloseTo(0.565685, 6);
  });

  it("supports in-place operation", () => {
    const input = new Float32Array([1, 2]);

    rmsNorm(input, tensor("weight", [2], new Float32Array([1, 1])), input);

    expect(input[0]).toBeCloseTo(0.632455, 6);
    expect(input[1]).toBeCloseTo(1.264911, 6);
  });

  it("rejects non-vector weight tensors", () => {
    expect(() =>
      rmsNorm(
        new Float32Array(2),
        tensor("weight", [1, 2], new Float32Array([1, 1])),
        new Float32Array(2),
      ),
    ).toThrow("weight must be rank 1");
  });

  it("rejects invalid epsilon values", () => {
    expect(() =>
      rmsNorm(
        new Float32Array(2),
        tensor("weight", [2], new Float32Array([1, 1])),
        new Float32Array(2),
        { epsilon: 0 },
      ),
    ).toThrow(RangeError);
  });

  it("rejects out-of-bounds spans", () => {
    expect(() =>
      rmsNorm(
        new Float32Array(2),
        tensor("weight", [2], new Float32Array([1, 1])),
        new Float32Array(2),
        { inputOffset: 1, featureSize: 2 },
      ),
    ).toThrow(RangeError);
  });
});

function tensor(name: string, shape: number[], data: Float32Array): TensorView {
  return {
    name,
    shape,
    byteOffset: 0,
    byteLength: data.byteLength,
    data,
  };
}

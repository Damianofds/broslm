import { describe, expect, it } from "vitest";
import type { TensorView } from "../../src/tensor";
import { layerNorm } from "../../src/primitives/layerNorm";

describe("layerNorm", () => {
  it("normalizes a feature vector and applies weight and bias", () => {
    const input = new Float32Array([1, 2, 3]);
    const weight: TensorView = {
      name: "weight",
      shape: [3],
      byteOffset: 0,
      byteLength: 12,
      data: new Float32Array([1, 1.5, 2]),
    };
    const bias: TensorView = {
      name: "bias",
      shape: [3],
      byteOffset: 0,
      byteLength: 12,
      data: new Float32Array([0, 0.5, -0.5]),
    };
    const output = new Float32Array(3);

    layerNorm(input, weight, bias, output, { epsilon: 1e-5 });

    expect(output[0]).toBeCloseTo(-1.2247357, 6);
    expect(output[1]).toBeCloseTo(0.5, 6);
    expect(output[2]).toBeCloseTo(1.9494714, 6);
  });

  it("respects offsets and explicit feature size", () => {
    const input = new Float32Array([99, 2, 4, 99]);
    const weight: TensorView = {
      name: "weight",
      shape: [2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([1, 1]),
    };
    const bias: TensorView = {
      name: "bias",
      shape: [2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([0, 0]),
    };
    const output = new Float32Array([7, 7, 7, 7]);

    layerNorm(input, weight, bias, output, {
      inputOffset: 1,
      outputOffset: 2,
      featureSize: 2,
      epsilon: 1e-5,
    });

    expect(output[0]).toBe(7);
    expect(output[1]).toBe(7);
    expect(output[2]).toBeCloseTo(-0.999995, 6);
    expect(output[3]).toBeCloseTo(0.999995, 6);
  });

  it("rejects non-vector weight tensors", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [1, 2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([1, 1]),
    };
    const bias: TensorView = {
      name: "bias",
      shape: [2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([0, 0]),
    };

    expect(() => layerNorm(new Float32Array(2), weight, bias, new Float32Array(2))).toThrow(
      "weight must be rank 1",
    );
  });

  it("rejects invalid epsilon values", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([1, 1]),
    };
    const bias: TensorView = {
      name: "bias",
      shape: [2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([0, 0]),
    };

    expect(() =>
      layerNorm(new Float32Array(2), weight, bias, new Float32Array(2), { epsilon: 0 }),
    ).toThrow(RangeError);
  });
});

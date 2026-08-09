import { describe, expect, it } from "vitest";
import type { TensorView } from "../../src/loader";
import { matrixVectorMultiply } from "../../src/primitives/matrixVectorMultiply";

describe("matrixVectorMultiply", () => {
  it("multiplies a row-major matrix by a vector", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [2, 3],
      byteOffset: 0,
      byteLength: 24,
      data: new Float32Array([1, 2, 3, 4, 5, 6]),
    };
    const output = new Float32Array(2);

    matrixVectorMultiply(weight, new Float32Array([10, 20, 30]), output);

    expect(Array.from(output)).toEqual([140, 320]);
  });

  it("adds Float32Array bias", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [2, 2],
      byteOffset: 0,
      byteLength: 16,
      data: new Float32Array([1, 0, 0, 1]),
    };
    const output = new Float32Array(2);

    matrixVectorMultiply(weight, new Float32Array([5, 7]), output, {
      bias: new Float32Array([0.5, -1]),
    });

    expect(Array.from(output)).toEqual([5.5, 6]);
  });

  it("adds TensorView bias and respects offsets", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [2, 2],
      byteOffset: 0,
      byteLength: 16,
      data: new Float32Array([2, 3, 4, 5]),
    };
    const bias: TensorView = {
      name: "bias",
      shape: [2],
      byteOffset: 0,
      byteLength: 8,
      data: new Float32Array([1, 2]),
    };
    const output = new Float32Array([9, 9, 9, 9]);

    matrixVectorMultiply(weight, new Float32Array([99, 10, 20]), output, {
      bias,
      inputOffset: 1,
      outputOffset: 2,
    });

    expect(Array.from(output)).toEqual([9, 9, 81, 142]);
  });

  it("rejects non-matrix weights", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [4],
      byteOffset: 0,
      byteLength: 16,
      data: new Float32Array([1, 2, 3, 4]),
    };

    expect(() => matrixVectorMultiply(weight, new Float32Array(2), new Float32Array(2))).toThrow(
      "weight must be rank 2",
    );
  });

  it("rejects bias with the wrong length", () => {
    const weight: TensorView = {
      name: "weight",
      shape: [2, 2],
      byteOffset: 0,
      byteLength: 16,
      data: new Float32Array([1, 2, 3, 4]),
    };

    expect(() =>
      matrixVectorMultiply(weight, new Float32Array(2), new Float32Array(2), {
        bias: new Float32Array([1]),
      }),
    ).toThrow("bias length is 1, expected 2");
  });
});

import { describe, expect, it } from "vitest";
import { elementwiseMultiply } from "../../src/primitives/elementwiseMultiply";

describe("elementwiseMultiply", () => {
  it("multiplies inputs elementwise", () => {
    const output = new Float32Array(3);

    elementwiseMultiply(new Float32Array([1, 2, 3]), new Float32Array([10, 20, 30]), output);

    expect(Array.from(output)).toEqual([10, 40, 90]);
  });

  it("respects separate left, right, and output offsets", () => {
    const output = new Float32Array([9, 9, 9, 9]);

    elementwiseMultiply(new Float32Array([99, 2, 3]), new Float32Array([10, 20, 99]), output, {
      leftOffset: 1,
      rightOffset: 0,
      outputOffset: 2,
      length: 2,
    });

    expect(Array.from(output)).toEqual([9, 9, 20, 60]);
  });

  it("allows zero-length spans", () => {
    const output = new Float32Array([1, 2]);

    elementwiseMultiply(new Float32Array([3, 4]), new Float32Array([5, 6]), output, {
      length: 0,
    });

    expect(Array.from(output)).toEqual([1, 2]);
  });

  it("supports in-place operation", () => {
    const left = new Float32Array([2, 3]);

    elementwiseMultiply(left, new Float32Array([10, 20]), left);

    expect(Array.from(left)).toEqual([20, 60]);
  });

  it("rejects out-of-bounds spans", () => {
    expect(() =>
      elementwiseMultiply(new Float32Array(2), new Float32Array(2), new Float32Array(2), {
        rightOffset: 1,
        length: 2,
      }),
    ).toThrow(RangeError);
  });
});

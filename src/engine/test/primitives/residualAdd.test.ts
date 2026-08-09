import { describe, expect, it } from "vitest";
import { residualAdd } from "../../src/primitives/residualAdd";

describe("residualAdd", () => {
  it("adds input and residual elementwise", () => {
    const output = new Float32Array(3);

    residualAdd(new Float32Array([1, 2, 3]), new Float32Array([10, 20, 30]), output);

    expect(Array.from(output)).toEqual([11, 22, 33]);
  });

  it("respects separate input, residual, and output offsets", () => {
    const output = new Float32Array([9, 9, 9, 9]);

    residualAdd(new Float32Array([99, 1, 2]), new Float32Array([10, 20, 99]), output, {
      inputOffset: 1,
      residualOffset: 0,
      outputOffset: 2,
      length: 2,
    });

    expect(Array.from(output)).toEqual([9, 9, 11, 22]);
  });

  it("allows zero-length spans", () => {
    const output = new Float32Array([1, 2]);

    residualAdd(new Float32Array([3, 4]), new Float32Array([5, 6]), output, { length: 0 });

    expect(Array.from(output)).toEqual([1, 2]);
  });

  it("rejects out-of-bounds spans", () => {
    expect(() =>
      residualAdd(new Float32Array(2), new Float32Array(2), new Float32Array(2), {
        residualOffset: 1,
        length: 2,
      }),
    ).toThrow(RangeError);
  });
});

import { describe, expect, it } from "vitest";
import { softmax } from "../../src/primitives/softmax";

describe("softmax", () => {
  it("normalizes logits into probabilities", () => {
    const output = new Float32Array(3);

    softmax(new Float32Array([1, 2, 3]), output);

    expect(output[0]).toBeCloseTo(0.09003057, 6);
    expect(output[1]).toBeCloseTo(0.24472847, 6);
    expect(output[2]).toBeCloseTo(0.66524096, 6);
    expect(output[0] + output[1] + output[2]).toBeCloseTo(1, 6);
  });

  it("is stable for large logits", () => {
    const output = new Float32Array(2);

    softmax(new Float32Array([1000, 1000]), output);

    expect(output[0]).toBeCloseTo(0.5, 6);
    expect(output[1]).toBeCloseTo(0.5, 6);
  });

  it("respects input and output offsets", () => {
    const output = new Float32Array([9, 9, 9, 9]);

    softmax(new Float32Array([99, 1, 2, 99]), output, {
      inputOffset: 1,
      outputOffset: 2,
      length: 2,
    });

    expect(output[0]).toBe(9);
    expect(output[1]).toBe(9);
    expect(output[2]).toBeCloseTo(0.26894142, 6);
    expect(output[3]).toBeCloseTo(0.73105858, 6);
  });

  it("leaves output untouched for zero-length spans", () => {
    const output = new Float32Array([1, 2]);

    softmax(new Float32Array([3, 4]), output, { length: 0 });

    expect(Array.from(output)).toEqual([1, 2]);
  });

  it("rejects out-of-bounds spans", () => {
    expect(() => softmax(new Float32Array(2), new Float32Array(2), { inputOffset: 1, length: 2 })).toThrow(
      RangeError,
    );
  });
});

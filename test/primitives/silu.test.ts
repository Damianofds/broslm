import { describe, expect, it } from "vitest";
import { silu } from "../../src/primitives/silu";

describe("silu", () => {
  it("applies the sigmoid linear unit activation", () => {
    const output = new Float32Array(4);

    silu(new Float32Array([-1, 0, 1, 2]), output);

    expect(output[0]).toBeCloseTo(-0.268941, 6);
    expect(output[1]).toBeCloseTo(0, 6);
    expect(output[2]).toBeCloseTo(0.731059, 6);
    expect(output[3]).toBeCloseTo(1.761594, 6);
  });

  it("respects input and output offsets", () => {
    const input = new Float32Array([99, -1, 1, 99]);
    const output = new Float32Array([7, 7, 7, 7]);

    silu(input, output, { inputOffset: 1, outputOffset: 2, length: 2 });

    expect(output[0]).toBe(7);
    expect(output[1]).toBe(7);
    expect(output[2]).toBeCloseTo(-0.268941, 6);
    expect(output[3]).toBeCloseTo(0.731059, 6);
  });

  it("supports in-place operation", () => {
    const input = new Float32Array([-1, 1]);

    silu(input, input);

    expect(input[0]).toBeCloseTo(-0.268941, 6);
    expect(input[1]).toBeCloseTo(0.731059, 6);
  });

  it("rejects spans outside the input", () => {
    expect(() =>
      silu(new Float32Array(2), new Float32Array(2), { inputOffset: 1, length: 2 }),
    ).toThrow(RangeError);
  });
});

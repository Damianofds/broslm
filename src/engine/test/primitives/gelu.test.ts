import { describe, expect, it } from "vitest";
import { gelu } from "../../src/primitives/gelu";

describe("gelu", () => {
  it("applies the GPT-Neo gelu_new approximation", () => {
    const input = new Float32Array([-1, 0, 1, 2]);
    const output = new Float32Array(4);

    gelu(input, output);

    expect(output[0]).toBeCloseTo(-0.158808, 6);
    expect(output[1]).toBeCloseTo(0, 6);
    expect(output[2]).toBeCloseTo(0.841192, 6);
    expect(output[3]).toBeCloseTo(1.954598, 6);
  });

  it("respects input and output offsets", () => {
    const input = new Float32Array([99, -1, 1, 99]);
    const output = new Float32Array([7, 7, 7, 7]);

    gelu(input, output, { inputOffset: 1, outputOffset: 2, length: 2 });

    expect(output[0]).toBe(7);
    expect(output[1]).toBe(7);
    expect(output[2]).toBeCloseTo(-0.158808, 6);
    expect(output[3]).toBeCloseTo(0.841192, 6);
  });

  it("rejects spans outside the input", () => {
    expect(() => gelu(new Float32Array(2), new Float32Array(2), { inputOffset: 1, length: 2 })).toThrow(
      RangeError,
    );
  });
});

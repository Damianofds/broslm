import { describe, expect, it } from "vitest";
import { applyRoPE } from "../../src/primitives/applyRoPE";

describe("applyRoPE", () => {
  it("leaves a head slice unchanged at position zero", () => {
    const output = new Float32Array(4);

    applyRoPE(new Float32Array([1, 2, 3, 4]), output, {
      position: 0,
      theta: 10000,
    });

    expect(Array.from(output)).toEqual([1, 2, 3, 4]);
  });

  it("applies Qwen-style half rotation at non-zero positions", () => {
    const output = new Float32Array(4);

    applyRoPE(new Float32Array([1, 2, 3, 4]), output, {
      position: 2,
      theta: 10000,
    });

    expect(output[0]).toBeCloseTo(-3.144039, 6);
    expect(output[1]).toBeCloseTo(1.919605, 6);
    expect(output[2]).toBeCloseTo(-0.339143, 6);
    expect(output[3]).toBeCloseTo(4.039197, 6);
  });

  it("respects input and output offsets", () => {
    const output = new Float32Array([7, 7, 7, 7, 7, 7]);

    applyRoPE(new Float32Array([99, 1, 2, 3, 4, 99]), output, {
      inputOffset: 1,
      outputOffset: 2,
      headDimension: 4,
      position: 0,
      theta: 10000,
    });

    expect(Array.from(output)).toEqual([7, 7, 1, 2, 3, 4]);
  });

  it("supports in-place operation", () => {
    const input = new Float32Array([1, 2, 3, 4]);

    applyRoPE(input, input, {
      position: 2,
      theta: 10000,
    });

    expect(input[0]).toBeCloseTo(-3.144039, 6);
    expect(input[1]).toBeCloseTo(1.919605, 6);
    expect(input[2]).toBeCloseTo(-0.339143, 6);
    expect(input[3]).toBeCloseTo(4.039197, 6);
  });

  it("rejects odd head dimensions", () => {
    expect(() =>
      applyRoPE(new Float32Array(3), new Float32Array(3), {
        headDimension: 3,
        position: 0,
        theta: 10000,
      }),
    ).toThrow(RangeError);
  });

  it("rejects invalid positions and theta values", () => {
    expect(() =>
      applyRoPE(new Float32Array(4), new Float32Array(4), {
        position: -1,
        theta: 10000,
      }),
    ).toThrow(RangeError);
    expect(() =>
      applyRoPE(new Float32Array(4), new Float32Array(4), {
        position: 0,
        theta: 0,
      }),
    ).toThrow(RangeError);
  });

  it("rejects out-of-bounds spans", () => {
    expect(() =>
      applyRoPE(new Float32Array(4), new Float32Array(4), {
        inputOffset: 1,
        headDimension: 4,
        position: 0,
        theta: 10000,
      }),
    ).toThrow(RangeError);
  });
});

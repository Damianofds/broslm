import { describe, expect, it } from "vitest";
import { sampleTokenFromLogits } from "../src/model";

describe("sampleTokenFromLogits", () => {
  it("falls back to argmax when temperature is zero", () => {
    const tokenId = sampleTokenFromLogits(new Float32Array([1, 4, 2]), {
      temperature: 0,
      topK: 3,
      random: () => 0.99,
    });

    expect(tokenId).toBe(1);
  });

  it("falls back to argmax when topK is one", () => {
    const tokenId = sampleTokenFromLogits(new Float32Array([1, 4, 2]), {
      temperature: 1,
      topK: 1,
      random: () => 0,
    });

    expect(tokenId).toBe(1);
  });

  it("samples from the probability mass after temperature scaling", () => {
    const logits = new Float32Array([0, 0]);

    expect(sampleTokenFromLogits(logits, { temperature: 1, topK: 2, random: () => 0.25 })).toBe(0);
    expect(sampleTokenFromLogits(logits, { temperature: 1, topK: 2, random: () => 0.75 })).toBe(1);
  });

  it("excludes tokens outside topK", () => {
    const tokenId = sampleTokenFromLogits(new Float32Array([10, 9, 8]), {
      temperature: 1,
      topK: 2,
      random: () => 0.999,
    });

    expect([0, 1]).toContain(tokenId);
  });
});

import { describe, expect, it } from "vitest";
import { argmax } from "../../src/primitives/argmax";

describe("argmax", () => {
  it("returns the index of the largest value", () => {
    expect(argmax(new Float32Array([1, 4, 2]))).toBe(1);
  });

  it("returns the first index when the largest value is tied", () => {
    expect(argmax(new Float32Array([3, 7, 7, 2]))).toBe(1);
  });

  it("handles negative values", () => {
    expect(argmax(new Float32Array([-10, -3, -7]))).toBe(1);
  });

  it("rejects empty inputs", () => {
    expect(() => argmax(new Float32Array())).toThrow("argmax requires at least one value");
  });
});

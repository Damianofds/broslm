import { describe, expect, it } from "vitest";
import type { TensorView } from "../../src/tensor";
import { embeddingLookup } from "../../src/primitives/embeddingLookup";

describe("embeddingLookup", () => {
  it("copies the row selected by token id", () => {
    const embedding: TensorView = {
      name: "embedding",
      shape: [3, 4],
      byteOffset: 0,
      byteLength: 48,
      data: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400]),
    };
    const output = new Float32Array(4);

    embeddingLookup(embedding, 1, output);

    expect(Array.from(output)).toEqual([10, 20, 30, 40]);
  });

  it("writes at the output offset", () => {
    const embedding: TensorView = {
      name: "embedding",
      shape: [2, 3],
      byteOffset: 0,
      byteLength: 24,
      data: new Float32Array([1, 2, 3, 4, 5, 6]),
    };
    const output = new Float32Array([9, 9, 9, 9, 9]);

    embeddingLookup(embedding, 0, output, 2);

    expect(Array.from(output)).toEqual([9, 9, 1, 2, 3]);
  });

  it("rejects token ids outside the table", () => {
    const embedding: TensorView = {
      name: "embedding",
      shape: [2, 2],
      byteOffset: 0,
      byteLength: 16,
      data: new Float32Array([1, 2, 3, 4]),
    };

    expect(() => embeddingLookup(embedding, 2, new Float32Array(2))).toThrow(RangeError);
  });

  it("rejects malformed embedding shapes", () => {
    const embedding: TensorView = {
      name: "embedding",
      shape: [4],
      byteOffset: 0,
      byteLength: 16,
      data: new Float32Array([1, 2, 3, 4]),
    };

    expect(() => embeddingLookup(embedding, 0, new Float32Array(2))).toThrow(
      "embedding must be rank 2",
    );
  });
});

/// <reference types="node" />

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createByteLevelBpeTokenizer } from "./tokenizer";

const tokenizerJsonPath = new URL("../public/tokenizer/tinystories-tokenizer.json", import.meta.url);

describe("ByteLevel BPE tokenizer", () => {
  const tokenizer = createByteLevelBpeTokenizer(
    JSON.parse(readFileSync(tokenizerJsonPath, "utf8")),
  );

  it("encodes known TinyStories prompts to GPT-2 token ids", () => {
    expect(tokenizer.encode("Once upon a time")).toEqual([7454, 2402, 257, 640]);
    expect(tokenizer.encode("The little girl found a")).toEqual([
      464,
      1310,
      2576,
      1043,
      257,
    ]);
  });

  it("decodes generated token ids back to text", () => {
    expect(tokenizer.decode([1263, 11, 2705, 18447, 13])).toBe(" big, soft blanket.");
    expect(tokenizer.decode([13, 1119, 2497, 257, 1263, 11, 39145, 3290, 13, 198])).toBe(
      ". They saw a big, fluffy dog.\n",
    );
  });

  it("round-trips plain prompt text", () => {
    const text = "Emma and Jack went to the park.";
    expect(tokenizer.decode(tokenizer.encode(text))).toBe(text);
  });
});

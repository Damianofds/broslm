/// <reference types="node" />

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  createByteLevelBpeTokenizer,
  createQwen2ByteLevelBpeTokenizer,
} from "./tokenizer";

const tokenizerJsonPath = new URL("./public/tokenizer/tinystories-tokenizer.json", import.meta.url);

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

  it("encodes and decodes Qwen2 GGUF-style chat control tokens literally", () => {
    const qwenTokenizer = createQwen2ByteLevelBpeTokenizer({
      tokens: ["a", "<|im_start|>", "<|im_end|>"],
      merges: [],
      tokenTypes: [1, 3, 3],
      eosTokenId: 2,
    });

    const tokenIds = qwenTokenizer.encode("<|im_start|>a<|im_end|>");

    expect(tokenIds).toEqual([1, 0, 2]);
    expect(qwenTokenizer.decode(tokenIds)).toBe("<|im_start|>a<|im_end|>");
    expect(qwenTokenizer.eosTokenId).toBe(2);
  });

  it("falls back to Qwen2 GGUF byte tokens when byte-level vocab entries are absent", () => {
    const qwenTokenizer = createQwen2ByteLevelBpeTokenizer({
      tokens: ["<0xC3>", "<0xA9>"],
      merges: [],
      tokenTypes: [6, 6],
      eosTokenId: null,
    });

    const tokenIds = qwenTokenizer.encode("é");

    expect(tokenIds).toEqual([0, 1]);
    expect(qwenTokenizer.decode(tokenIds)).toBe("é");
  });
});

import { describe, expect, it } from "vitest";
import { createQwen2ByteLevelBpeTokenizer } from "./tokenizer";

describe("ByteLevel BPE tokenizer", () => {
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

import { describe, expect, it } from "vitest";
import { BroslmError } from "../../src/errors";
import { formatQwen2Prompt } from "../../src/qwen2/chat";
import type { ChatMessage } from "../../src/types";

describe("Qwen2 ChatML formatting", () => {
  it("converts a string to a user message while preserving legacy normalization", () => {
    expect(formatQwen2Prompt("first\r\n\r\nsecond")).toBe(
      "<|im_start|>system\n" +
      "You are a helpful assistant.<|im_end|>\n" +
      "<|im_start|>user\nfirst\nsecond<|im_end|>\n" +
      "<|im_start|>assistant\n",
    );
  });

  it("formats a structured multi-turn conversation and preserves content", () => {
    expect(formatQwen2Prompt([
      { role: "system", content: "Answer as TypeScript." },
      { role: "user", content: "Write a function." },
      { role: "assistant", content: "```ts\nfunction first() {}\n```" },
      { role: "user", content: "Keep this gap:\n\nDone." },
    ])).toBe(
      "<|im_start|>system\nAnswer as TypeScript.<|im_end|>\n" +
      "<|im_start|>user\nWrite a function.<|im_end|>\n" +
      "<|im_start|>assistant\n```ts\nfunction first() {}\n```<|im_end|>\n" +
      "<|im_start|>user\nKeep this gap:\n\nDone.<|im_end|>\n" +
      "<|im_start|>assistant\n",
    );
  });

  it("adds the default system message to a structured user-only prompt", () => {
    expect(formatQwen2Prompt([{ role: "user", content: "Hello" }])).toContain(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    );
  });

  it.each([
    ["an empty conversation", []],
    ["an unsupported role", [{ role: "tool", content: "result" }]],
    ["empty content", [{ role: "user", content: "" }]],
    ["a misplaced system message", [
      { role: "user", content: "hello" },
      { role: "system", content: "late" },
      { role: "user", content: "again" },
    ]],
    ["no user message", [{ role: "system", content: "instructions" }]],
    ["an assistant final message", [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hello" },
    ]],
  ])("rejects %s", (_label, messages) => {
    expect(() => formatQwen2Prompt(messages as readonly ChatMessage[])).toThrow(BroslmError);
  });
});

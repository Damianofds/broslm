import { describe, expect, it } from "vitest";
import {
  defaultModelId,
  formatPromptForModel,
  modelCatalog,
  modelLoadSteps,
  stepIndexForProgressStage,
  visibleGeneratedTextForModel,
} from "./modelCatalog";

describe("modelCatalog", () => {
  it("keeps TinyStories as the default UI model", () => {
    expect(defaultModelId).toBe("tinystories");
    expect(modelCatalog.tinystories.baseUrl).toContain("/models/output_20260726_105535/");
    expect(modelCatalog.tinystories.tokenizerUrl).toContain(
      "/tokenizer/tinystories-tokenizer.json",
    );
  });

  it("points Qwen at the documented local GGUF path", () => {
    expect(modelCatalog.qwen.baseUrl).toContain("/models/qwen2.5-0.5b-instruct-q4_0/");
    expect(modelCatalog.qwen.ggufPath).toBe("model.gguf");
    expect(modelCatalog.qwen.ggufFallbackUrls).toEqual([
      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
    ]);
  });

  it("formats Qwen prompts with a hidden chat template", () => {
    expect(formatPromptForModel("tinystories", "Once upon a time")).toBe("Once upon a time");
    expect(formatPromptForModel("qwen", "Hello")).toBe(
      "<|im_start|>system\n" +
        "You are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\nHello<|im_end|>\n" +
        "<|im_start|>assistant\n",
    );
  });

  it("strips Qwen control tokens from visible generated text", () => {
    expect(visibleGeneratedTextForModel("tinystories", "<|im_end|>")).toBe("<|im_end|>");
    expect(visibleGeneratedTextForModel("qwen", "Done<|im_end|>")).toBe("Done");
  });

  it("maps model-specific progress stages onto visible step indexes", () => {
    expect(stepIndexForProgressStage("weights-download-progress", modelLoadSteps.tinystories)).toBe(3);
    expect(stepIndexForProgressStage("gguf-download-progress", modelLoadSteps.qwen)).toBe(0);
    expect(stepIndexForProgressStage("gguf-parsed", modelLoadSteps.qwen)).toBe(2);
  });
});

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
  it("keeps the optimized Qwen2.5 GGUF as the default UI model", () => {
    expect(defaultModelId).toBe("qwen");
    expect(Object.keys(modelCatalog).sort()).toEqual(["qwen", "qwen_cpu_small"]);
    expect(modelCatalog.qwen.backendPolicy.webgpu).toBe("required");
    expect(modelCatalog.qwen.backendPolicy.cpuFallback).toBe(false);
  });

  it("points Qwen directly at the official Hugging Face GGUF", () => {
    expect(modelCatalog.qwen.baseUrl).toContain("/models/qwen2.5-0.5b-instruct-q4_0/");
    expect(modelCatalog.qwen.ggufPath).toBe(
      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    expect(modelCatalog.qwen.ggufFallbackUrls).toBeUndefined();
  });

  it("adds a CPU-only Qwen IQ1_S model", () => {
    expect(modelCatalog.qwen_cpu_small.baseUrl).toContain("/models/qwen2.5-0.5b-instruct-iq1_s/");
    expect(modelCatalog.qwen_cpu_small.ggufPath).toBe(
      "https://huggingface.co/legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF/resolve/main/Qwen2.5-0.5B-Instruct.IQ1_S.gguf",
    );
    expect(modelCatalog.qwen_cpu_small.backendPolicy).toMatchObject({
      defaultPreference: "cpu",
      webgpu: "unsupported",
      cpuFallback: true,
    });
  });

  it("formats Qwen prompts with a hidden chat template", () => {
    expect(formatPromptForModel("qwen", "Hello")).toBe(
      "<|im_start|>system\n" +
        "You are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\nHello<|im_end|>\n" +
        "<|im_start|>assistant\n",
    );
    expect(formatPromptForModel("qwen_cpu_small", "Hello")).toBe(formatPromptForModel("qwen", "Hello"));
  });

  it("strips Qwen control tokens from visible generated text", () => {
    expect(visibleGeneratedTextForModel("qwen", "Done<|im_end|>")).toBe("Done");
    expect(visibleGeneratedTextForModel("qwen_cpu_small", "Done<|im_end|>")).toBe("Done");
  });

  it("maps model-specific progress stages onto visible step indexes", () => {
    expect(stepIndexForProgressStage("gguf-download-progress", modelLoadSteps.qwen)).toBe(0);
    expect(stepIndexForProgressStage("gguf-parsed", modelLoadSteps.qwen)).toBe(2);
    expect(stepIndexForProgressStage("ready", modelLoadSteps.qwen_cpu_small)).toBe(4);
  });
});

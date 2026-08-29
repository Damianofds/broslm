import { describe, expect, it } from "vitest";
import {
  createBroslm,
  defaultModelId,
  modelCatalog,
  modelOptions,
} from "../src/browser";

describe("public package API", () => {
  it("exposes the two stable Qwen model profiles", () => {
    expect(defaultModelId).toBe("qwen");
    expect(modelOptions.map((model) => model.id)).toEqual(["qwen", "qwen_cpu_small"]);
    expect(modelCatalog.qwen.backend).toBe("webgpu");
    expect(modelCatalog.qwen_cpu_small.backend).toBe("cpu");
  });

  it("creates a disposable browser client without touching WebGPU", () => {
    const client = createBroslm();
    expect(client.state).toBe("idle");
    client.dispose();
    expect(client.state).toBe("disposed");
  });
});

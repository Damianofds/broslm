import { describe, expect, it } from "vitest";
import {
  createBroslm,
  defaultModelId,
  modelCatalog,
  modelOptions,
} from "../src/browser";

describe("public package API", () => {
  it("exposes only the Qwen2.5 0.5B WebGPU profile", () => {
    expect(defaultModelId).toBe("qwen");
    expect(modelOptions.map((model) => model.id)).toEqual(["qwen"]);
    expect(modelCatalog.qwen.backend).toBe("webgpu");
    expect(modelCatalog.qwen.quantization).toBe("Q4_0");
  });

  it("creates a disposable browser client without touching WebGPU", () => {
    const client = createBroslm();
    expect(client.state).toBe("idle");
    client.dispose();
    expect(client.state).toBe("disposed");
  });
});

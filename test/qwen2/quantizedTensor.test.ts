import { describe, expect, it } from "vitest";
import {
  createQwenTensorView,
  isFloat32TensorView,
} from "../../src/qwen2/quantizedTensor";
import {
  GGML_TYPE_F32,
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q8_0,
  type GgufTensorInfo,
} from "../../src/qwen2/gguf";

describe("Qwen2.5 WebGPU tensor views", () => {
  it("accepts only the tensor formats used by the Q4_0 profile", () => {
    const buffer = new ArrayBuffer(256);
    const f32 = createQwenTensorView(buffer, tensorInfo("norm", [2], GGML_TYPE_F32, 8));
    const q4 = createQwenTensorView(buffer, tensorInfo("q4", [32, 1], GGML_TYPE_Q4_0, 18));
    const q8 = createQwenTensorView(buffer, tensorInfo("q8", [32, 1], GGML_TYPE_Q8_0, 34));

    expect(isFloat32TensorView(f32)).toBe(true);
    expect(isFloat32TensorView(q4)).toBe(false);
    expect(isFloat32TensorView(q8)).toBe(false);
    expect(q4).toMatchObject({ type: "q4_0", shape: [1, 32] });
    expect(q8).toMatchObject({ type: "q8_0", shape: [1, 32] });
  });

  it("rejects every other GGUF tensor format", () => {
    expect(() =>
      createQwenTensorView(new ArrayBuffer(32), tensorInfo("unsupported", [2], 19, 2)),
    ).toThrow("Unsupported Qwen2.5 0.5B WebGPU tensor type");
  });
});

function tensorInfo(
  name: string,
  dimensions: number[],
  type: number,
  byteLength: number,
): GgufTensorInfo {
  return {
    name,
    dimensions,
    type,
    offset: 0,
    byteOffset: 0,
    byteLength,
  };
}

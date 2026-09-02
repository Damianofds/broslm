import type { TensorView } from "../tensor";
import {
  GGML_TYPE_F32,
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q8_0,
  type GgufTensorInfo,
} from "./gguf";

export type QwenTensorType = "f32" | "q4_0" | "q8_0";

export interface QuantizedTensorView {
  name: string;
  shape: readonly number[];
  byteOffset: number;
  byteLength: number;
  type: Exclude<QwenTensorType, "f32">;
  data: Uint8Array;
}

export type QwenTensorView = TensorView | QuantizedTensorView;

export function createQwenTensorView(
  weightsBuffer: ArrayBuffer,
  tensor: GgufTensorInfo,
): QwenTensorView {
  const shape = logicalShape(tensor.dimensions);
  if (tensor.type === GGML_TYPE_F32) {
    return {
      name: tensor.name,
      shape: Object.freeze(shape),
      byteOffset: tensor.byteOffset,
      byteLength: tensor.byteLength,
      data: new Float32Array(weightsBuffer, tensor.byteOffset, tensor.byteLength / 4),
    };
  }

  const type = tensor.type === GGML_TYPE_Q4_0
    ? "q4_0"
    : tensor.type === GGML_TYPE_Q8_0
      ? "q8_0"
      : null;
  if (!type) {
    throw new Error(
      `Unsupported Qwen2.5 0.5B WebGPU tensor type ${tensor.type} for ${tensor.name}`,
    );
  }

  return {
    name: tensor.name,
    shape: Object.freeze(shape),
    byteOffset: tensor.byteOffset,
    byteLength: tensor.byteLength,
    type,
    data: new Uint8Array(weightsBuffer, tensor.byteOffset, tensor.byteLength),
  };
}

export function isFloat32TensorView(tensor: QwenTensorView): tensor is TensorView {
  return tensor.data instanceof Float32Array;
}

function logicalShape(dimensions: readonly number[]): number[] {
  return dimensions.length === 2
    ? [dimensions[1] ?? 0, dimensions[0] ?? 0]
    : [...dimensions];
}

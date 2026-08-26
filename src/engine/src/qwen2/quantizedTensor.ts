import type { TensorView } from "../tensor";
import { matrixVectorMultiply } from "../primitives/matrixVectorMultiply";
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

  const type = tensor.type === GGML_TYPE_Q4_0 ? "q4_0" : tensor.type === GGML_TYPE_Q8_0 ? "q8_0" : null;
  if (!type) {
    throw new Error(`Unsupported Qwen tensor type ${tensor.type} for ${tensor.name}`);
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

export function matrixVectorMultiplyQwen(
  weight: QwenTensorView,
  input: Float32Array,
  output: Float32Array,
  options: {
    bias?: TensorView | Float32Array;
    inputOffset?: number;
    outputOffset?: number;
  } = {},
): void {
  if (isFloat32TensorView(weight)) {
    matrixVectorMultiply(weight, input, output, options);
    return;
  }

  const [outputSize, inputSize] = requireMatrixShape(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const bias = options.bias ? resolveBias(options.bias, outputSize) : null;
  validateSpan("input", input.length, inputOffset, inputSize);
  validateSpan("output", output.length, outputOffset, outputSize);

  const rowByteLength = quantizedRowByteLength(weight.type, inputSize);
  for (let row = 0; row < outputSize; row += 1) {
    let sum = bias?.[row] ?? 0;
    const rowOffset = row * rowByteLength;
    if (weight.type === "q4_0") {
      sum += dotQ4_0Row(weight.data, rowOffset, input, inputOffset, inputSize);
    } else {
      sum += dotQ8_0Row(weight.data, rowOffset, input, inputOffset, inputSize);
    }
    output[outputOffset + row] = sum;
  }
}

export function embeddingLookupQwen(
  embedding: QwenTensorView,
  tokenId: number,
  output: Float32Array,
  outputOffset = 0,
): void {
  const [entryCount, embeddingSize] = requireMatrixShape(embedding, "embedding");
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= entryCount) {
    throw new RangeError(`tokenId must be an integer in [0, ${entryCount}), got ${tokenId}`);
  }
  validateSpan("output", output.length, outputOffset, embeddingSize);

  if (isFloat32TensorView(embedding)) {
    const sourceOffset = tokenId * embeddingSize;
    output.set(embedding.data.subarray(sourceOffset, sourceOffset + embeddingSize), outputOffset);
    return;
  }

  const rowByteLength = quantizedRowByteLength(embedding.type, embeddingSize);
  const rowOffset = tokenId * rowByteLength;
  if (embedding.type === "q4_0") {
    dequantizeQ4_0Row(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else {
    dequantizeQ8_0Row(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  }
}

export function dequantizeQ4_0Row(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("q4_0", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    sourceOffset += 2;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[sourceOffset + packedIndex] ?? 0;
      output[outputOffset + base + packedIndex] = ((packed & 0x0f) - 8) * scale;
      output[outputOffset + base + 16 + packedIndex] = ((packed >> 4) - 8) * scale;
    }
    sourceOffset += 16;
  }
}

export function dequantizeQ8_0Row(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("q8_0", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    sourceOffset += 2;
    for (let index = 0; index < 32; index += 1) {
      output[outputOffset + base + index] = toSignedInt8(data[sourceOffset + index] ?? 0) * scale;
    }
    sourceOffset += 32;
  }
}

export function float16ToFloat32(lowByte: number, highByte: number): number {
  const half = lowByte | (highByte << 8);
  const sign = (half & 0x8000) ? -1 : 1;
  const exponent = (half >> 10) & 0x1f;
  const fraction = half & 0x03ff;

  if (exponent === 0) {
    return sign * (fraction === 0 ? 0 : Math.pow(2, -14) * (fraction / 1024));
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

export function isFloat32TensorView(tensor: QwenTensorView): tensor is TensorView {
  return tensor.data instanceof Float32Array;
}

function dotQ4_0Row(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("q4_0", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    sourceOffset += 2;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[sourceOffset + packedIndex] ?? 0;
      sum += ((packed & 0x0f) - 8) * scale * (input[inputOffset + base + packedIndex] ?? 0);
      sum += ((packed >> 4) - 8) * scale * (input[inputOffset + base + 16 + packedIndex] ?? 0);
    }
    sourceOffset += 16;
  }
  return sum;
}

function dotQ8_0Row(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("q8_0", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    sourceOffset += 2;
    for (let index = 0; index < 32; index += 1) {
      sum += toSignedInt8(data[sourceOffset + index] ?? 0) * scale * (input[inputOffset + base + index] ?? 0);
    }
    sourceOffset += 32;
  }
  return sum;
}

function logicalShape(dimensions: readonly number[]): number[] {
  if (dimensions.length === 2) {
    return [dimensions[1] ?? 0, dimensions[0] ?? 0];
  }
  return [...dimensions];
}

function requireMatrixShape(tensor: QwenTensorView, name: string): [number, number] {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }
  const outputSize = tensor.shape[0] ?? 0;
  const inputSize = tensor.shape[1] ?? 0;
  if (outputSize <= 0 || inputSize <= 0) {
    throw new Error(`${name} dimensions must be positive, got [${outputSize}, ${inputSize}]`);
  }
  return [outputSize, inputSize];
}

function resolveBias(bias: TensorView | Float32Array, expectedLength: number): Float32Array {
  const values = bias instanceof Float32Array ? bias : bias.data;
  if (values.length !== expectedLength) {
    throw new Error(`bias length is ${values.length}, expected ${expectedLength}`);
  }
  return values;
}

function quantizedRowByteLength(type: Exclude<QwenTensorType, "f32">, inputSize: number): number {
  requireQuantizedLength(type, inputSize);
  return type === "q4_0" ? (inputSize / 32) * 18 : (inputSize / 32) * 34;
}

function requireQuantizedLength(type: Exclude<QwenTensorType, "f32">, length: number): void {
  if (!Number.isInteger(length) || length <= 0 || length % 32 !== 0) {
    throw new Error(`${type} length must be a positive multiple of 32, got ${length}`);
  }
}

function toSignedInt8(value: number): number {
  return value > 127 ? value - 256 : value;
}

function validateSpan(name: string, bufferLength: number, offset: number, length: number): void {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new RangeError(`${name} offset must be a non-negative integer, got ${offset}`);
  }
  if (offset + length > bufferLength) {
    throw new RangeError(
      `${name} span is out of bounds: need ${offset + length} values, got ${bufferLength}`,
    );
  }
}

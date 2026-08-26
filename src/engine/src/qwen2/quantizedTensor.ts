import type { TensorView } from "../tensor";
import { embeddingLookupGpu } from "../primitives/embeddingLookup";
import { matrixVectorMultiply } from "../primitives/matrixVectorMultiply";
import { matrixVectorMultiplyGpu } from "../primitives/matrixVectorMultiply";
import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";
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

export async function matrixVectorMultiplyQwenGpu(
  runtime: WebGpuRuntime,
  weight: QwenTensorView,
  input: Float32Array,
  options: {
    bias?: TensorView | Float32Array;
    inputOffset?: number;
  } = {},
): Promise<Float32Array> {
  if (isFloat32TensorView(weight)) {
    return matrixVectorMultiplyGpu(runtime, weight, input, options);
  }

  const [outputSize, inputSize] = requireMatrixShape(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const bias = options.bias ? resolveBias(options.bias, outputSize) : new Float32Array(outputSize);
  validateSpan("input", input.length, inputOffset, inputSize);

  const rowByteLength = quantizedRowByteLength(weight.type, inputSize);
  const weightBuffer = createStorageBuffer(runtime, weight.data);
  const inputBuffer = createStorageBuffer(runtime, input);
  const biasBuffer = createStorageBuffer(runtime, bias);
  const outputBuffer = createStorageBuffer(runtime, outputSize * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([
      inputSize,
      inputOffset,
      outputSize,
      rowByteLength,
      weight.type === "q4_0" ? 4 : 8,
      0,
      0,
      0,
    ]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      qwenQuantizedMatvecShader,
      [
        { binding: 0, resource: { buffer: weightBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(outputSize / 64)],
    );
    return readFloat32Buffer(runtime, outputBuffer, outputSize);
  } finally {
    destroyBuffers(weightBuffer, inputBuffer, biasBuffer, outputBuffer, paramsBuffer);
  }
}

const qwenQuantizedMatvecShader = `
struct Params {
  inputSize: u32,
  inputOffset: u32,
  outputSize: u32,
  rowByteLength: u32,
  quantType: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn byteAt(byteOffset: u32) -> u32 {
  let word = weightWords[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByteAt(byteOffset: u32) -> f32 {
  let value = byteAt(byteOffset);
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let row = globalId.x;
  if (row >= params.outputSize) {
    return;
  }

  var sum = bias[row];
  var sourceOffset = row * params.rowByteLength;
  for (var base = 0u; base < params.inputSize; base = base + 32u) {
    let scale = f16ToF32(byteAt(sourceOffset), byteAt(sourceOffset + 1u));
    sourceOffset = sourceOffset + 2u;

    if (params.quantType == 4u) {
      for (var packedIndex = 0u; packedIndex < 16u; packedIndex = packedIndex + 1u) {
        let packed = byteAt(sourceOffset + packedIndex);
        let low = f32(i32(packed & 0x0fu) - 8) * scale;
        let high = f32(i32((packed >> 4u) & 0x0fu) - 8) * scale;
        sum = sum + low * input[params.inputOffset + base + packedIndex];
        sum = sum + high * input[params.inputOffset + base + 16u + packedIndex];
      }
      sourceOffset = sourceOffset + 16u;
    } else {
      for (var index = 0u; index < 32u; index = index + 1u) {
        sum = sum + signedByteAt(sourceOffset + index) * scale * input[params.inputOffset + base + index];
      }
      sourceOffset = sourceOffset + 32u;
    }
  }
  output[row] = sum;
}
`;

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

export async function embeddingLookupQwenGpu(
  runtime: WebGpuRuntime,
  embedding: QwenTensorView,
  tokenId: number,
): Promise<Float32Array> {
  const [entryCount, embeddingSize] = requireMatrixShape(embedding, "embedding");
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= entryCount) {
    throw new RangeError(`tokenId must be an integer in [0, ${entryCount}), got ${tokenId}`);
  }

  if (isFloat32TensorView(embedding)) {
    return embeddingLookupGpu(runtime, embedding, tokenId);
  }

  const rowByteLength = quantizedRowByteLength(embedding.type, embeddingSize);
  const dataBuffer = createStorageBuffer(runtime, embedding.data);
  const outputBuffer = createStorageBuffer(runtime, embeddingSize * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([
      tokenId,
      embeddingSize,
      rowByteLength,
      embedding.type === "q4_0" ? 4 : 8,
    ]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      qwenQuantizedEmbeddingShader,
      [
        { binding: 0, resource: { buffer: dataBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(embeddingSize / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, embeddingSize);
  } finally {
    destroyBuffers(dataBuffer, outputBuffer, paramsBuffer);
  }
}

const qwenQuantizedEmbeddingShader = `
struct Params {
  tokenId: u32,
  embeddingSize: u32,
  rowByteLength: u32,
  quantType: u32,
}

@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn byteAt(byteOffset: u32) -> u32 {
  let word = words[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByteAt(byteOffset: u32) -> f32 {
  let value = byteAt(byteOffset);
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.embeddingSize) {
    return;
  }

  let rowOffset = params.tokenId * params.rowByteLength;
  let block = index / 32u;
  let blockIndex = index % 32u;
  let blockOffset = rowOffset + block * select(18u, 34u, params.quantType == 8u);
  let scale = f16ToF32(byteAt(blockOffset), byteAt(blockOffset + 1u));

  if (params.quantType == 4u) {
    let packed = byteAt(blockOffset + 2u + (blockIndex % 16u));
    let quantized = select((packed >> 4u) & 0x0fu, packed & 0x0fu, blockIndex < 16u);
    output[index] = f32(i32(quantized) - 8) * scale;
  } else {
    output[index] = signedByteAt(blockOffset + 2u + blockIndex) * scale;
  }
}
`;

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

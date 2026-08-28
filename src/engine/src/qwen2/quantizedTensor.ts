import type { TensorView } from "../tensor";
import { embeddingLookupGpu } from "../primitives/embeddingLookup";
import { matrixVectorMultiply } from "../primitives/matrixVectorMultiply";
import { matrixVectorMultiplyGpu } from "../primitives/matrixVectorMultiply";
import {
  createStorageBuffer,
  createStaticStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";
import {
  GGML_TYPE_F32,
  GGML_TYPE_IQ1_M,
  GGML_TYPE_IQ1_S,
  GGML_TYPE_IQ4_NL,
  GGML_TYPE_Q2_K,
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q5_0,
  GGML_TYPE_Q5_1,
  GGML_TYPE_Q8_0,
  type GgufTensorInfo,
} from "./gguf";
import { iq1sGridGpu } from "./iq1sGrid";
import { iq4nlValues } from "./iq4nlValues";

export type QwenTensorType =
  | "f32"
  | "q4_0"
  | "q5_0"
  | "q5_1"
  | "q8_0"
  | "q2_k"
  | "iq1_s"
  | "iq4_nl"
  | "iq1_m";

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

  const type =
    tensor.type === GGML_TYPE_Q4_0
      ? "q4_0"
      : tensor.type === GGML_TYPE_Q5_0
        ? "q5_0"
        : tensor.type === GGML_TYPE_Q5_1
        ? "q5_1"
        : tensor.type === GGML_TYPE_Q8_0
          ? "q8_0"
          : tensor.type === GGML_TYPE_Q2_K
            ? "q2_k"
            : tensor.type === GGML_TYPE_IQ1_S
              ? "iq1_s"
              : tensor.type === GGML_TYPE_IQ4_NL
                ? "iq4_nl"
              : tensor.type === GGML_TYPE_IQ1_M
                ? "iq1_m"
                : null;
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
    } else if (weight.type === "q5_0") {
      sum += dotQ5_0Row(weight.data, rowOffset, input, inputOffset, inputSize);
    } else if (weight.type === "q5_1") {
      sum += dotQ5_1Row(weight.data, rowOffset, input, inputOffset, inputSize);
    } else if (weight.type === "q8_0") {
      sum += dotQ8_0Row(weight.data, rowOffset, input, inputOffset, inputSize);
    } else if (weight.type === "q2_k") {
      sum += dotQ2_KRow(weight.data, rowOffset, input, inputOffset, inputSize);
    } else if (weight.type === "iq4_nl") {
      sum += dotIQ4_NLRow(weight.data, rowOffset, input, inputOffset, inputSize);
    } else if (weight.type === "iq1_m") {
      sum += dotIQ1_MRow(weight.data, rowOffset, input, inputOffset, inputSize);
    } else {
      sum += dotIQ1_SRow(weight.data, rowOffset, input, inputOffset, inputSize);
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
  assertWebGpuSupportedQuantizedTensor(weight);

  const [outputSize, inputSize] = requireMatrixShape(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const bias = options.bias ? resolveBias(options.bias, outputSize) : new Float32Array(outputSize);
  const biasIsStaticTensor = options.bias !== undefined && !(options.bias instanceof Float32Array);
  validateSpan("input", input.length, inputOffset, inputSize);

  const rowByteLength = quantizedRowByteLength(weight.type, inputSize);
  const weightBuffer = createStaticStorageBuffer(runtime, weight.data);
  const inputBuffer = createStorageBuffer(runtime, input);
  const biasBuffer = biasIsStaticTensor
    ? createStaticStorageBuffer(runtime, bias)
    : createStorageBuffer(runtime, bias);
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
    destroyBuffers(inputBuffer, biasIsStaticTensor ? undefined : biasBuffer, outputBuffer, paramsBuffer);
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
  } else if (embedding.type === "q5_0") {
    dequantizeQ5_0Row(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else if (embedding.type === "q5_1") {
    dequantizeQ5_1Row(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else if (embedding.type === "q8_0") {
    dequantizeQ8_0Row(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else if (embedding.type === "q2_k") {
    dequantizeQ2_KRow(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else if (embedding.type === "iq4_nl") {
    dequantizeIQ4_NLRow(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else if (embedding.type === "iq1_m") {
    dequantizeIQ1_MRow(embedding.data, rowOffset, output, outputOffset, embeddingSize);
  } else {
    dequantizeIQ1_SRow(embedding.data, rowOffset, output, outputOffset, embeddingSize);
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
  assertWebGpuSupportedQuantizedTensor(embedding);

  const rowByteLength = quantizedRowByteLength(embedding.type, embeddingSize);
  const dataBuffer = createStaticStorageBuffer(runtime, embedding.data);
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
    destroyBuffers(outputBuffer, paramsBuffer);
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

export function dequantizeIQ4_NLRow(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("iq4_nl", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const quantOffset = sourceOffset + 2;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[quantOffset + packedIndex] ?? 0;
      output[outputOffset + base + packedIndex] =
        scale * (iq4nlValues[packed & 0x0f] ?? 0);
      output[outputOffset + base + 16 + packedIndex] =
        scale * (iq4nlValues[packed >> 4] ?? 0);
    }
    sourceOffset += 18;
  }
}

export function dequantizeQ5_1Row(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("q5_1", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const min = float16ToFloat32(data[sourceOffset + 2] ?? 0, data[sourceOffset + 3] ?? 0);
    const highBits = readUint32Le(data, sourceOffset + 4);
    const qsOffset = sourceOffset + 8;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[qsOffset + packedIndex] ?? 0;
      const lowHighBit = ((highBits >> packedIndex) << 4) & 0x10;
      const highHighBit = (highBits >> (packedIndex + 12)) & 0x10;
      output[outputOffset + base + packedIndex] = ((packed & 0x0f) | lowHighBit) * scale + min;
      output[outputOffset + base + 16 + packedIndex] = ((packed >> 4) | highHighBit) * scale + min;
    }
    sourceOffset += 24;
  }
}

export function dequantizeQ5_0Row(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("q5_0", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const highBits = readUint32Le(data, sourceOffset + 2);
    const qsOffset = sourceOffset + 6;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[qsOffset + packedIndex] ?? 0;
      const lowHighBit = ((highBits >> packedIndex) << 4) & 0x10;
      const highHighBit = (highBits >> (packedIndex + 12)) & 0x10;
      output[outputOffset + base + packedIndex] = (((packed & 0x0f) | lowHighBit) - 16) * scale;
      output[outputOffset + base + 16 + packedIndex] = (((packed >> 4) | highHighBit) - 16) * scale;
    }
    sourceOffset += 22;
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

export function dequantizeQ2_KRow(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("q2_k", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 256) {
    const remaining = Math.min(256, length - base);
    const scaleBase = sourceOffset;
    const quantBase = sourceOffset + 16;
    const d = float16ToFloat32(data[sourceOffset + 80] ?? 0, data[sourceOffset + 81] ?? 0);
    const dmin = float16ToFloat32(data[sourceOffset + 82] ?? 0, data[sourceOffset + 83] ?? 0);
    let scaleIndex = 0;
    let quantOffset = quantBase;
    for (let blockBase = 0; blockBase < 256; blockBase += 128) {
      let shift = 0;
      for (let group = 0; group < 4; group += 1) {
        const firstScale = data[scaleBase + scaleIndex++] ?? 0;
        const firstD = d * (firstScale & 0x0f);
        const firstMin = dmin * (firstScale >> 4);
        for (let index = 0; index < 16; index += 1) {
          const quantized = ((data[quantOffset + index] ?? 0) >> shift) & 3;
          const outputIndex = blockBase + group * 32 + index;
          if (outputIndex < remaining) {
            output[outputOffset + base + outputIndex] = firstD * quantized - firstMin;
          }
        }

        const secondScale = data[scaleBase + scaleIndex++] ?? 0;
        const secondD = d * (secondScale & 0x0f);
        const secondMin = dmin * (secondScale >> 4);
        for (let index = 0; index < 16; index += 1) {
          const quantized = ((data[quantOffset + 16 + index] ?? 0) >> shift) & 3;
          const outputIndex = blockBase + group * 32 + 16 + index;
          if (outputIndex < remaining) {
            output[outputOffset + base + outputIndex] = secondD * quantized - secondMin;
          }
        }
        shift += 2;
      }
      quantOffset += 32;
    }
    sourceOffset += 84;
  }
}

export function dequantizeIQ1_SRow(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("iq1_s", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 256) {
    const remaining = Math.min(256, length - base);
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const qsOffset = sourceOffset + 2;
    const qhOffset = sourceOffset + 34;
    for (let subblock = 0; subblock < 8; subblock += 1) {
      const qh = readUint16Le(data, qhOffset + subblock * 2);
      const subblockScale = scale * (2 * ((qh >> 12) & 7) + 1);
      const delta = qh & 0x8000 ? -1.125 : -0.875;
      for (let lane = 0; lane < 4; lane += 1) {
        const gridIndex = (data[qsOffset + 4 * subblock + lane] ?? 0) | (((qh >> (3 * lane)) & 7) << 8);
        const grid = iq1sGridGpu[gridIndex] ?? 0;
        for (let item = 0; item < 8; item += 1) {
          const shift = 8 * (item % 4) + (item >= 4 ? 4 : 0);
          const quantized = (grid >> shift) & 0x0f;
          const outputIndex = 32 * subblock + 8 * lane + item;
          if (outputIndex < remaining) {
            output[outputOffset + base + outputIndex] = subblockScale * (quantized + delta);
          }
        }
      }
    }
    sourceOffset += 50;
  }
}

export function dequantizeIQ1_MRow(
  data: Uint8Array,
  rowOffset: number,
  output: Float32Array,
  outputOffset: number,
  length: number,
): void {
  requireQuantizedLength("iq1_m", length);
  let sourceOffset = rowOffset;
  for (let base = 0; base < length; base += 256) {
    const remaining = Math.min(256, length - base);
    const qsOffset = sourceOffset;
    const qhOffset = sourceOffset + 32;
    const scalesOffset = sourceOffset + 48;
    const scale = iq1MScale(data, scalesOffset);
    for (let subblock = 0; subblock < 8; subblock += 1) {
      const scaleWord = readUint16Le(data, scalesOffset + 2 * Math.floor(subblock / 2));
      const shiftBase = 6 * (subblock % 2);
      const firstScale = scale * (2 * ((scaleWord >> shiftBase) & 0x07) + 1);
      const secondScale = scale * (2 * ((scaleWord >> (shiftBase + 3)) & 0x07) + 1);
      const qh0 = data[qhOffset + 2 * subblock] ?? 0;
      const qh1 = data[qhOffset + 2 * subblock + 1] ?? 0;
      const indexes = [
        (data[qsOffset + 4 * subblock] ?? 0) | ((qh0 << 8) & 0x700),
        (data[qsOffset + 4 * subblock + 1] ?? 0) | ((qh0 << 4) & 0x700),
        (data[qsOffset + 4 * subblock + 2] ?? 0) | ((qh1 << 8) & 0x700),
        (data[qsOffset + 4 * subblock + 3] ?? 0) | ((qh1 << 4) & 0x700),
      ];
      const deltas = [
        qh0 & 0x08 ? -1.125 : -0.875,
        qh0 & 0x80 ? -1.125 : -0.875,
        qh1 & 0x08 ? -1.125 : -0.875,
        qh1 & 0x80 ? -1.125 : -0.875,
      ];
      for (let lane = 0; lane < 4; lane += 1) {
        const laneScale = lane < 2 ? firstScale : secondScale;
        const grid = iq1sGridGpu[indexes[lane] ?? 0] ?? 0;
        for (let item = 0; item < 8; item += 1) {
          const shift = 8 * (item % 4) + (item >= 4 ? 4 : 0);
          const quantized = (grid >> shift) & 0x0f;
          const outputIndex = 32 * subblock + 8 * lane + item;
          if (outputIndex < remaining) {
            output[outputOffset + base + outputIndex] =
              laneScale * (quantized + (deltas[lane] ?? 0));
          }
        }
      }
    }
    sourceOffset += 56;
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

function dotQ5_1Row(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("q5_1", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const min = float16ToFloat32(data[sourceOffset + 2] ?? 0, data[sourceOffset + 3] ?? 0);
    const highBits = readUint32Le(data, sourceOffset + 4);
    const qsOffset = sourceOffset + 8;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[qsOffset + packedIndex] ?? 0;
      const lowHighBit = ((highBits >> packedIndex) << 4) & 0x10;
      const highHighBit = (highBits >> (packedIndex + 12)) & 0x10;
      sum += (((packed & 0x0f) | lowHighBit) * scale + min) * (input[inputOffset + base + packedIndex] ?? 0);
      sum += (((packed >> 4) | highHighBit) * scale + min) * (input[inputOffset + base + 16 + packedIndex] ?? 0);
    }
    sourceOffset += 24;
  }
  return sum;
}

function dotIQ4_NLRow(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("iq4_nl", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const quantOffset = sourceOffset + 2;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[quantOffset + packedIndex] ?? 0;
      sum +=
        scale * (iq4nlValues[packed & 0x0f] ?? 0) *
        (input[inputOffset + base + packedIndex] ?? 0);
      sum +=
        scale * (iq4nlValues[packed >> 4] ?? 0) *
        (input[inputOffset + base + 16 + packedIndex] ?? 0);
    }
    sourceOffset += 18;
  }
  return sum;
}

function dotQ5_0Row(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("q5_0", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 32) {
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const highBits = readUint32Le(data, sourceOffset + 2);
    const qsOffset = sourceOffset + 6;
    for (let packedIndex = 0; packedIndex < 16; packedIndex += 1) {
      const packed = data[qsOffset + packedIndex] ?? 0;
      const lowHighBit = ((highBits >> packedIndex) << 4) & 0x10;
      const highHighBit = (highBits >> (packedIndex + 12)) & 0x10;
      sum +=
        (((packed & 0x0f) | lowHighBit) - 16) *
        scale *
        (input[inputOffset + base + packedIndex] ?? 0);
      sum +=
        (((packed >> 4) | highHighBit) - 16) *
        scale *
        (input[inputOffset + base + 16 + packedIndex] ?? 0);
    }
    sourceOffset += 22;
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

function dotQ2_KRow(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("q2_k", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 256) {
    const remaining = Math.min(256, length - base);
    const scaleBase = sourceOffset;
    const quantBase = sourceOffset + 16;
    const d = float16ToFloat32(data[sourceOffset + 80] ?? 0, data[sourceOffset + 81] ?? 0);
    const dmin = float16ToFloat32(data[sourceOffset + 82] ?? 0, data[sourceOffset + 83] ?? 0);
    let scaleIndex = 0;
    let quantOffset = quantBase;
    for (let blockBase = 0; blockBase < 256; blockBase += 128) {
      let shift = 0;
      for (let group = 0; group < 4; group += 1) {
        const firstScale = data[scaleBase + scaleIndex++] ?? 0;
        const firstD = d * (firstScale & 0x0f);
        const firstMin = dmin * (firstScale >> 4);
        for (let index = 0; index < 16; index += 1) {
          const column = base + blockBase + group * 32 + index;
          const quantized = ((data[quantOffset + index] ?? 0) >> shift) & 3;
          if (column - base < remaining) {
            sum += (firstD * quantized - firstMin) * (input[inputOffset + column] ?? 0);
          }
        }

        const secondScale = data[scaleBase + scaleIndex++] ?? 0;
        const secondD = d * (secondScale & 0x0f);
        const secondMin = dmin * (secondScale >> 4);
        for (let index = 0; index < 16; index += 1) {
          const column = base + blockBase + group * 32 + 16 + index;
          const quantized = ((data[quantOffset + 16 + index] ?? 0) >> shift) & 3;
          if (column - base < remaining) {
            sum += (secondD * quantized - secondMin) * (input[inputOffset + column] ?? 0);
          }
        }
        shift += 2;
      }
      quantOffset += 32;
    }
    sourceOffset += 84;
  }
  return sum;
}

function dotIQ1_SRow(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("iq1_s", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 256) {
    const remaining = Math.min(256, length - base);
    const scale = float16ToFloat32(data[sourceOffset] ?? 0, data[sourceOffset + 1] ?? 0);
    const qsOffset = sourceOffset + 2;
    const qhOffset = sourceOffset + 34;
    for (let subblock = 0; subblock < 8; subblock += 1) {
      const qh = readUint16Le(data, qhOffset + subblock * 2);
      const subblockScale = scale * (2 * ((qh >> 12) & 7) + 1);
      const delta = qh & 0x8000 ? -1.125 : -0.875;
      for (let lane = 0; lane < 4; lane += 1) {
        const gridIndex = (data[qsOffset + 4 * subblock + lane] ?? 0) | (((qh >> (3 * lane)) & 7) << 8);
        const grid = iq1sGridGpu[gridIndex] ?? 0;
        for (let item = 0; item < 8; item += 1) {
          const shift = 8 * (item % 4) + (item >= 4 ? 4 : 0);
          const quantized = (grid >> shift) & 0x0f;
          const columnOffset = 32 * subblock + 8 * lane + item;
          if (columnOffset < remaining) {
            sum +=
              subblockScale *
              (quantized + delta) *
              (input[inputOffset + base + columnOffset] ?? 0);
          }
        }
      }
    }
    sourceOffset += 50;
  }
  return sum;
}

function dotIQ1_MRow(
  data: Uint8Array,
  rowOffset: number,
  input: Float32Array,
  inputOffset: number,
  length: number,
): number {
  requireQuantizedLength("iq1_m", length);
  let sourceOffset = rowOffset;
  let sum = 0;
  for (let base = 0; base < length; base += 256) {
    const remaining = Math.min(256, length - base);
    const qsOffset = sourceOffset;
    const qhOffset = sourceOffset + 32;
    const scalesOffset = sourceOffset + 48;
    const scale = iq1MScale(data, scalesOffset);
    for (let subblock = 0; subblock < 8; subblock += 1) {
      const scaleWord = readUint16Le(data, scalesOffset + 2 * Math.floor(subblock / 2));
      const shiftBase = 6 * (subblock % 2);
      const firstScale = scale * (2 * ((scaleWord >> shiftBase) & 0x07) + 1);
      const secondScale = scale * (2 * ((scaleWord >> (shiftBase + 3)) & 0x07) + 1);
      const qh0 = data[qhOffset + 2 * subblock] ?? 0;
      const qh1 = data[qhOffset + 2 * subblock + 1] ?? 0;
      const indexes = [
        (data[qsOffset + 4 * subblock] ?? 0) | ((qh0 << 8) & 0x700),
        (data[qsOffset + 4 * subblock + 1] ?? 0) | ((qh0 << 4) & 0x700),
        (data[qsOffset + 4 * subblock + 2] ?? 0) | ((qh1 << 8) & 0x700),
        (data[qsOffset + 4 * subblock + 3] ?? 0) | ((qh1 << 4) & 0x700),
      ];
      const deltas = [
        qh0 & 0x08 ? -1.125 : -0.875,
        qh0 & 0x80 ? -1.125 : -0.875,
        qh1 & 0x08 ? -1.125 : -0.875,
        qh1 & 0x80 ? -1.125 : -0.875,
      ];
      for (let lane = 0; lane < 4; lane += 1) {
        const laneScale = lane < 2 ? firstScale : secondScale;
        const grid = iq1sGridGpu[indexes[lane] ?? 0] ?? 0;
        for (let item = 0; item < 8; item += 1) {
          const column = base + 32 * subblock + 8 * lane + item;
          const shift = 8 * (item % 4) + (item >= 4 ? 4 : 0);
          const quantized = (grid >> shift) & 0x0f;
          if (column - base < remaining) {
            sum += laneScale * (quantized + (deltas[lane] ?? 0)) * (input[inputOffset + column] ?? 0);
          }
        }
      }
    }
    sourceOffset += 56;
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
  if (type === "q4_0") {
    return (inputSize / 32) * 18;
  }
  if (type === "q5_0") {
    return (inputSize / 32) * 22;
  }
  if (type === "q5_1") {
    return (inputSize / 32) * 24;
  }
  if (type === "q8_0") {
    return (inputSize / 32) * 34;
  }
  if (type === "q2_k") {
    return Math.ceil(inputSize / 256) * 84;
  }
  if (type === "iq4_nl") {
    return (inputSize / 32) * 18;
  }
  if (type === "iq1_m") {
    return Math.ceil(inputSize / 256) * 56;
  }
  return Math.ceil(inputSize / 256) * 50;
}

function requireQuantizedLength(type: Exclude<QwenTensorType, "f32">, length: number): void {
  const blockSize = type === "iq1_s" || type === "iq1_m" || type === "q2_k" ? 256 : 32;
  const requiresExactBlocks = blockSize === 32;
  if (
    !Number.isInteger(length) ||
    length <= 0 ||
    (requiresExactBlocks && length % blockSize !== 0)
  ) {
    throw new Error(`${type} length must be a positive multiple of ${blockSize}, got ${length}`);
  }
}

function assertWebGpuSupportedQuantizedTensor(tensor: QuantizedTensorView): void {
  if (tensor.type !== "q4_0" && tensor.type !== "q8_0") {
    throw new Error(`${tensor.name} uses ${quantizationLabel(tensor.type)} quantization, which is only supported on the CPU backend.`);
  }
}

function quantizationLabel(type: Exclude<QwenTensorType, "f32">): string {
  return type.toUpperCase();
}

function readUint16Le(data: Uint8Array, offset: number): number {
  return (data[offset] ?? 0) | ((data[offset + 1] ?? 0) << 8);
}

function readUint32Le(data: Uint8Array, offset: number): number {
  return (
    (data[offset] ?? 0) |
    ((data[offset + 1] ?? 0) << 8) |
    ((data[offset + 2] ?? 0) << 16) |
    ((data[offset + 3] ?? 0) << 24)
  ) >>> 0;
}

function iq1MScale(data: Uint8Array, scalesOffset: number): number {
  // GGML packs the FP16 scale from the high nibble of four little-endian
  // scale words. Keep this reconstruction identical to iq1m_scale_t.
  const scaleWord0 = readUint16Le(data, scalesOffset);
  const scaleWord1 = readUint16Le(data, scalesOffset + 2);
  const scaleWord2 = readUint16Le(data, scalesOffset + 4);
  const scaleWord3 = readUint16Le(data, scalesOffset + 6);
  const half =
    (scaleWord0 >> 12) |
    ((scaleWord1 >> 8) & 0x00f0) |
    ((scaleWord2 >> 4) & 0x0f00) |
    (scaleWord3 & 0xf000);
  return float16ToFloat32(half & 0xff, (half >> 8) & 0xff);
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

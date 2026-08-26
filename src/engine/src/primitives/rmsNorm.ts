import type { TensorView } from "../tensor";
import {
  createStorageBuffer,
  createStaticStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export function rmsNorm(
  input: Float32Array,
  weight: TensorView,
  output: Float32Array,
  options: { inputOffset?: number; outputOffset?: number; featureSize?: number; epsilon?: number } = {},
): void {
  const featureSize = options.featureSize ?? requireVectorLength(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const epsilon = options.epsilon ?? 1e-6;

  requireVectorLength(weight, "weight", featureSize);
  validateSpan("input", input.length, inputOffset, featureSize);
  validateSpan("output", output.length, outputOffset, featureSize);
  if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError(`epsilon must be a positive finite number, got ${epsilon}`);
  }

  const scale = 1 / Math.sqrt(calculateMeanSquare(input, inputOffset, featureSize) + epsilon);
  for (let index = 0; index < featureSize; index += 1) {
    output[outputOffset + index] =
      (input[inputOffset + index] ?? 0) * scale * (weight.data[index] ?? 0);
  }
}

export async function rmsNormGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  weight: TensorView,
  options: { inputOffset?: number; featureSize?: number; epsilon?: number } = {},
): Promise<Float32Array> {
  const featureSize = options.featureSize ?? requireVectorLength(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const epsilon = options.epsilon ?? 1e-6;

  requireVectorLength(weight, "weight", featureSize);
  validateSpan("input", input.length, inputOffset, featureSize);
  if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError(`epsilon must be a positive finite number, got ${epsilon}`);
  }

  const inputBuffer = createStorageBuffer(runtime, input);
  const weightBuffer = createStaticStorageBuffer(runtime, weight.data);
  const outputBuffer = createStorageBuffer(runtime, featureSize * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Float32Array([epsilon, inputOffset, featureSize, 0]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      rmsNormShader,
      [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
      [1],
    );
    return readFloat32Buffer(runtime, outputBuffer, featureSize);
  } finally {
    destroyBuffers(inputBuffer, outputBuffer, paramsBuffer);
  }
}

const rmsNormShader = `
struct Params {
  epsilon: f32,
  inputOffset: f32,
  featureSize: f32,
  _padding: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
  let inputOffset = u32(params.inputOffset);
  let featureSize = u32(params.featureSize);
  var meanSquare = 0.0;
  for (var index = 0u; index < featureSize; index = index + 1u) {
    let value = input[inputOffset + index];
    meanSquare = meanSquare + value * value;
  }
  meanSquare = meanSquare / f32(featureSize);
  let scale = inverseSqrt(meanSquare + params.epsilon);

  for (var index = 0u; index < featureSize; index = index + 1u) {
    output[index] = input[inputOffset + index] * scale * weight[index];
  }
}
`;

function calculateMeanSquare(input: Float32Array, offset: number, length: number): number {
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    const value = input[offset + index] ?? 0;
    sum += value * value;
  }
  return sum / length;
}

function requireVectorLength(tensor: TensorView, name: string, expectedLength?: number): number {
  if (tensor.shape.length !== 1) {
    throw new Error(`${name} must be rank 1, got shape [${tensor.shape.join(", ")}]`);
  }

  const length = tensor.shape[0] ?? 0;
  if (length <= 0) {
    throw new Error(`${name} length must be positive, got ${length}`);
  }
  if (tensor.data.length !== length) {
    throw new Error(`${name} data length does not match shape [${length}]`);
  }
  if (expectedLength !== undefined && length !== expectedLength) {
    throw new Error(`${name} length is ${length}, expected ${expectedLength}`);
  }

  return length;
}

function validateSpan(name: string, bufferLength: number, offset: number, length: number): void {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new RangeError(`${name} offset must be a non-negative integer, got ${offset}`);
  }
  if (!Number.isInteger(length) || length <= 0) {
    throw new RangeError(`${name} length must be a positive integer, got ${length}`);
  }
  if (offset + length > bufferLength) {
    throw new RangeError(
      `${name} span is out of bounds: need ${offset + length} values, got ${bufferLength}`,
    );
  }
}

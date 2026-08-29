import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export function softmax(
  input: Float32Array,
  output: Float32Array,
  options: { inputOffset?: number; outputOffset?: number; length?: number } = {},
): void {
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const length = options.length ?? input.length - inputOffset;

  validateSpan("input", input.length, inputOffset, length);
  validateSpan("output", output.length, outputOffset, length);
  if (length === 0) {
    return;
  }

  const maxValue = findMax(input, inputOffset, length);
  let sum = 0;

  for (let index = 0; index < length; index += 1) {
    const exponent = Math.exp((input[inputOffset + index] ?? 0) - maxValue);
    output[outputOffset + index] = exponent;
    sum += exponent;
  }

  if (sum === 0 || !Number.isFinite(sum)) {
    throw new Error("softmax normalization sum is not finite");
  }

  const inverseSum = 1 / sum;
  for (let index = 0; index < length; index += 1) {
    output[outputOffset + index] = (output[outputOffset + index] ?? 0) * inverseSum;
  }
}

export async function softmaxGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  options: { inputOffset?: number; length?: number } = {},
): Promise<Float32Array> {
  const inputOffset = options.inputOffset ?? 0;
  const length = options.length ?? input.length - inputOffset;
  validateSpan("input", input.length, inputOffset, length);
  if (length === 0) {
    return new Float32Array(0);
  }

  const inputBuffer = createStorageBuffer(runtime, input);
  const outputBuffer = createStorageBuffer(runtime, length * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([inputOffset, length, 0, 0]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      softmaxShader,
      [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
      [1],
    );
    return readFloat32Buffer(runtime, outputBuffer, length);
  } finally {
    destroyBuffers(inputBuffer, outputBuffer, paramsBuffer);
  }
}

const softmaxShader = `
struct Params {
  inputOffset: u32,
  length: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
  var maxValue = -3.4028234663852886e38;
  for (var index = 0u; index < params.length; index = index + 1u) {
    maxValue = max(maxValue, input[params.inputOffset + index]);
  }

  var sum = 0.0;
  for (var index = 0u; index < params.length; index = index + 1u) {
    let value = exp(input[params.inputOffset + index] - maxValue);
    output[index] = value;
    sum = sum + value;
  }

  for (var index = 0u; index < params.length; index = index + 1u) {
    output[index] = output[index] / sum;
  }
}
`;

function findMax(input: Float32Array, offset: number, length: number): number {
  let maxValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < length; index += 1) {
    maxValue = Math.max(maxValue, input[offset + index] ?? Number.NEGATIVE_INFINITY);
  }
  return maxValue;
}

function validateSpan(name: string, bufferLength: number, offset: number, length: number): void {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new RangeError(`${name} offset must be a non-negative integer, got ${offset}`);
  }
  if (!Number.isInteger(length) || length < 0) {
    throw new RangeError(`${name} length must be a non-negative integer, got ${length}`);
  }
  if (offset + length > bufferLength) {
    throw new RangeError(
      `${name} span is out of bounds: need ${offset + length} values, got ${bufferLength}`,
    );
  }
}

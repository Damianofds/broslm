import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export function gelu(
  input: Float32Array,
  output: Float32Array,
  options: { inputOffset?: number; outputOffset?: number; length?: number } = {},
): void {
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const length = options.length ?? input.length - inputOffset;
  validateSpan("input", input.length, inputOffset, length);
  validateSpan("output", output.length, outputOffset, length);

  for (let index = 0; index < length; index += 1) {
    const value = input[inputOffset + index] ?? 0;
    output[outputOffset + index] = geluNew(value);
  }
}

export async function geluGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  options: { inputOffset?: number; length?: number } = {},
): Promise<Float32Array> {
  const inputOffset = options.inputOffset ?? 0;
  const length = options.length ?? input.length - inputOffset;
  validateSpan("input", input.length, inputOffset, length);

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
      geluShader,
      [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(length / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, length);
  } finally {
    destroyBuffers(inputBuffer, outputBuffer, paramsBuffer);
  }
}

const geluShader = `
struct Params {
  inputOffset: u32,
  length: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  let value = input[params.inputOffset + index];
  output[index] = 0.5 * value * (1.0 + tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)));
}
`;

function geluNew(value: number): number {
  return (
    0.5 *
    value *
    (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value * value * value)))
  );
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

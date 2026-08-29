import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export function residualAdd(
  input: Float32Array,
  residual: Float32Array,
  output: Float32Array,
  options: {
    inputOffset?: number;
    residualOffset?: number;
    outputOffset?: number;
    length?: number;
  } = {},
): void {
  const inputOffset = options.inputOffset ?? 0;
  const residualOffset = options.residualOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const length = options.length ?? input.length - inputOffset;

  validateSpan("input", input.length, inputOffset, length);
  validateSpan("residual", residual.length, residualOffset, length);
  validateSpan("output", output.length, outputOffset, length);

  for (let index = 0; index < length; index += 1) {
    output[outputOffset + index] =
      (input[inputOffset + index] ?? 0) + (residual[residualOffset + index] ?? 0);
  }
}

export async function residualAddGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  residual: Float32Array,
  options: {
    inputOffset?: number;
    residualOffset?: number;
    length?: number;
  } = {},
): Promise<Float32Array> {
  const inputOffset = options.inputOffset ?? 0;
  const residualOffset = options.residualOffset ?? 0;
  const length = options.length ?? input.length - inputOffset;

  validateSpan("input", input.length, inputOffset, length);
  validateSpan("residual", residual.length, residualOffset, length);

  const inputBuffer = createStorageBuffer(runtime, input);
  const residualBuffer = createStorageBuffer(runtime, residual);
  const outputBuffer = createStorageBuffer(runtime, length * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([inputOffset, residualOffset, length, 0]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      residualAddShader,
      [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: residualBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(length / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, length);
  } finally {
    destroyBuffers(inputBuffer, residualBuffer, outputBuffer, paramsBuffer);
  }
}

const residualAddShader = `
struct Params {
  inputOffset: u32,
  residualOffset: u32,
  length: u32,
  _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  output[index] = input[params.inputOffset + index] + residual[params.residualOffset + index];
}
`;

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

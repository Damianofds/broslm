import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export function elementwiseMultiply(
  left: Float32Array,
  right: Float32Array,
  output: Float32Array,
  options: {
    leftOffset?: number;
    rightOffset?: number;
    outputOffset?: number;
    length?: number;
  } = {},
): void {
  const leftOffset = options.leftOffset ?? 0;
  const rightOffset = options.rightOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const length = options.length ?? left.length - leftOffset;

  validateSpan("left", left.length, leftOffset, length);
  validateSpan("right", right.length, rightOffset, length);
  validateSpan("output", output.length, outputOffset, length);

  for (let index = 0; index < length; index += 1) {
    output[outputOffset + index] =
      (left[leftOffset + index] ?? 0) * (right[rightOffset + index] ?? 0);
  }
}

export async function elementwiseMultiplyGpu(
  runtime: WebGpuRuntime,
  left: Float32Array,
  right: Float32Array,
  options: {
    leftOffset?: number;
    rightOffset?: number;
    length?: number;
  } = {},
): Promise<Float32Array> {
  const leftOffset = options.leftOffset ?? 0;
  const rightOffset = options.rightOffset ?? 0;
  const length = options.length ?? left.length - leftOffset;

  validateSpan("left", left.length, leftOffset, length);
  validateSpan("right", right.length, rightOffset, length);

  const leftBuffer = createStorageBuffer(runtime, left);
  const rightBuffer = createStorageBuffer(runtime, right);
  const outputBuffer = createStorageBuffer(runtime, length * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([leftOffset, rightOffset, length, 0]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      elementwiseMultiplyShader,
      [
        { binding: 0, resource: { buffer: leftBuffer } },
        { binding: 1, resource: { buffer: rightBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(length / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, length);
  } finally {
    destroyBuffers(leftBuffer, rightBuffer, outputBuffer, paramsBuffer);
  }
}

const elementwiseMultiplyShader = `
struct Params {
  leftOffset: u32,
  rightOffset: u32,
  length: u32,
  _padding: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  output[index] = left[params.leftOffset + index] * right[params.rightOffset + index];
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

import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

export function applyRoPE(
  input: Float32Array,
  output: Float32Array,
  options: {
    position: number;
    theta: number;
    inputOffset?: number;
    outputOffset?: number;
    headDimension?: number;
  },
): void {
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const headDimension = options.headDimension ?? input.length - inputOffset;

  validatePosition(options.position);
  validateTheta(options.theta);
  validateHeadDimension(headDimension);
  validateSpan("input", input.length, inputOffset, headDimension);
  validateSpan("output", output.length, outputOffset, headDimension);

  const halfDimension = headDimension / 2;
  for (let pairIndex = 0; pairIndex < halfDimension; pairIndex += 1) {
    const angle =
      options.position / Math.pow(options.theta, (2 * pairIndex) / headDimension);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const firstValue = input[inputOffset + pairIndex] ?? 0;
    const secondValue = input[inputOffset + halfDimension + pairIndex] ?? 0;

    output[outputOffset + pairIndex] = firstValue * cosine - secondValue * sine;
    output[outputOffset + halfDimension + pairIndex] = secondValue * cosine + firstValue * sine;
  }
}

export async function applyRoPEGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  options: {
    position: number;
    theta: number;
    inputOffset?: number;
    headDimension?: number;
  },
): Promise<Float32Array> {
  const inputOffset = options.inputOffset ?? 0;
  const headDimension = options.headDimension ?? input.length - inputOffset;

  validatePosition(options.position);
  validateTheta(options.theta);
  validateHeadDimension(headDimension);
  validateSpan("input", input.length, inputOffset, headDimension);

  const inputBuffer = createStorageBuffer(runtime, input);
  const outputBuffer = createStorageBuffer(runtime, headDimension * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Float32Array([options.position, options.theta, inputOffset, headDimension]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      applyRoPEShader,
      [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil((headDimension / 2) / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, headDimension);
  } finally {
    destroyBuffers(inputBuffer, outputBuffer, paramsBuffer);
  }
}

const applyRoPEShader = `
struct Params {
  position: f32,
  theta: f32,
  inputOffset: f32,
  headDimension: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let pairIndex = globalId.x;
  let inputOffset = u32(params.inputOffset);
  let headDimension = u32(params.headDimension);
  let halfDimension = headDimension / 2u;
  if (pairIndex >= halfDimension) {
    return;
  }

  let angle = params.position / pow(params.theta, (2.0 * f32(pairIndex)) / params.headDimension);
  let cosine = cos(angle);
  let sine = sin(angle);
  let firstValue = input[inputOffset + pairIndex];
  let secondValue = input[inputOffset + halfDimension + pairIndex];
  output[pairIndex] = firstValue * cosine - secondValue * sine;
  output[halfDimension + pairIndex] = secondValue * cosine + firstValue * sine;
}
`;

function validatePosition(position: number): void {
  if (!Number.isInteger(position) || position < 0) {
    throw new RangeError(`position must be a non-negative integer, got ${position}`);
  }
}

function validateTheta(theta: number): void {
  if (typeof theta !== "number" || !Number.isFinite(theta) || theta <= 0) {
    throw new RangeError(`theta must be a positive finite number, got ${theta}`);
  }
}

function validateHeadDimension(headDimension: number): void {
  if (!Number.isInteger(headDimension) || headDimension <= 0) {
    throw new RangeError(`headDimension must be a positive integer, got ${headDimension}`);
  }
  if (headDimension % 2 !== 0) {
    throw new RangeError(`headDimension must be even, got ${headDimension}`);
  }
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

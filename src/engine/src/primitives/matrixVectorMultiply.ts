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

export function matrixVectorMultiply(
  weight: TensorView,
  input: Float32Array,
  output: Float32Array,
  options: {
    bias?: TensorView | Float32Array;
    inputOffset?: number;
    outputOffset?: number;
  } = {},
): void {
  const [outputSize, inputSize] = requireMatrixShape(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const bias = options.bias ? resolveBias(options.bias, outputSize) : null;

  validateSpan("input", input.length, inputOffset, inputSize);
  validateSpan("output", output.length, outputOffset, outputSize);

  for (let row = 0; row < outputSize; row += 1) {
    let sum = bias?.[row] ?? 0;
    const weightOffset = row * inputSize;

    for (let column = 0; column < inputSize; column += 1) {
      sum += (weight.data[weightOffset + column] ?? 0) * (input[inputOffset + column] ?? 0);
    }

    output[outputOffset + row] = sum;
  }
}

export async function matrixVectorMultiplyGpu(
  runtime: WebGpuRuntime,
  weight: TensorView,
  input: Float32Array,
  options: {
    bias?: TensorView | Float32Array;
    inputOffset?: number;
    outputOffset?: number;
  } = {},
): Promise<Float32Array> {
  const [outputSize, inputSize] = requireMatrixShape(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const bias = options.bias ? resolveBias(options.bias, outputSize) : new Float32Array(outputSize);
  const biasIsStaticTensor = options.bias !== undefined && !(options.bias instanceof Float32Array);

  validateSpan("input", input.length, inputOffset, inputSize);

  const output = createStorageBuffer(runtime, outputSize * Float32Array.BYTES_PER_ELEMENT);
  const weightBuffer = createStaticStorageBuffer(runtime, weight.data);
  const inputBuffer = createStorageBuffer(runtime, input);
  const biasBuffer = biasIsStaticTensor
    ? createStaticStorageBuffer(runtime, bias)
    : createStorageBuffer(runtime, bias);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([inputSize, inputOffset, outputSize, 0]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      matrixVectorMultiplyShader,
      [
        { binding: 0, resource: { buffer: weightBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: output } },
      ],
      [Math.ceil(outputSize / 64)],
    );
    return readFloat32Buffer(runtime, output, outputSize);
  } finally {
    destroyBuffers(output, inputBuffer, biasIsStaticTensor ? undefined : biasBuffer, paramsBuffer);
  }
}

const matrixVectorMultiplyShader = `
struct Params {
  inputSize: u32,
  inputOffset: u32,
  outputSize: u32,
  _padding: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let row = globalId.x;
  if (row >= params.outputSize) {
    return;
  }

  var sum = bias[row];
  let weightOffset = row * params.inputSize;
  for (var column = 0u; column < params.inputSize; column = column + 1u) {
    sum = sum + weight[weightOffset + column] * input[params.inputOffset + column];
  }
  output[row] = sum;
}
`;

function requireMatrixShape(tensor: TensorView, name: string): [number, number] {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }

  const outputSize = tensor.shape[0] ?? 0;
  const inputSize = tensor.shape[1] ?? 0;
  if (outputSize <= 0 || inputSize <= 0) {
    throw new Error(`${name} dimensions must be positive, got [${outputSize}, ${inputSize}]`);
  }
  if (tensor.data.length !== outputSize * inputSize) {
    throw new Error(`${name} data length does not match shape [${outputSize}, ${inputSize}]`);
  }

  return [outputSize, inputSize];
}

function resolveBias(bias: TensorView | Float32Array, expectedLength: number): Float32Array {
  const values = bias instanceof Float32Array ? bias : bias.data;
  if (!(values instanceof Float32Array)) {
    throw new Error("bias must resolve to a Float32Array");
  }
  if (values.length !== expectedLength) {
    throw new Error(`bias length is ${values.length}, expected ${expectedLength}`);
  }
  if (!(bias instanceof Float32Array) && (bias.shape.length !== 1 || bias.shape[0] !== expectedLength)) {
    throw new Error(`bias must be rank 1 with length ${expectedLength}`);
  }

  return values;
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

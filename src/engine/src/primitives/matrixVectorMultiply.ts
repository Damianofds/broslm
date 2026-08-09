import type { TensorView } from "../loader";

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

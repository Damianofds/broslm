import type { TensorView } from "../tensor";

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

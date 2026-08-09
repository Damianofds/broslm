import type { TensorView } from "../loader";

export function layerNorm(
  input: Float32Array,
  weight: TensorView,
  bias: TensorView,
  output: Float32Array,
  options: { inputOffset?: number; outputOffset?: number; featureSize?: number; epsilon?: number } = {},
): void {
  const featureSize = options.featureSize ?? requireVectorLength(weight, "weight");
  const inputOffset = options.inputOffset ?? 0;
  const outputOffset = options.outputOffset ?? 0;
  const epsilon = options.epsilon ?? 1e-5;

  requireVectorLength(weight, "weight", featureSize);
  requireVectorLength(bias, "bias", featureSize);
  validateSpan("input", input.length, inputOffset, featureSize);
  validateSpan("output", output.length, outputOffset, featureSize);
  if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError(`epsilon must be a positive finite number, got ${epsilon}`);
  }

  const mean = calculateMean(input, inputOffset, featureSize);
  const variance = calculateVariance(input, inputOffset, featureSize, mean);
  const scale = 1 / Math.sqrt(variance + epsilon);

  for (let index = 0; index < featureSize; index += 1) {
    const normalized = ((input[inputOffset + index] ?? 0) - mean) * scale;
    output[outputOffset + index] = normalized * (weight.data[index] ?? 0) + (bias.data[index] ?? 0);
  }
}

function calculateMean(input: Float32Array, offset: number, length: number): number {
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    sum += input[offset + index] ?? 0;
  }
  return sum / length;
}

function calculateVariance(input: Float32Array, offset: number, length: number, mean: number): number {
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    const centered = (input[offset + index] ?? 0) - mean;
    sum += centered * centered;
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

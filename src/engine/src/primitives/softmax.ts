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

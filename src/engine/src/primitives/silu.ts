export function silu(
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
    output[outputOffset + index] = value / (1 + Math.exp(-value));
  }
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

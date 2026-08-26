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

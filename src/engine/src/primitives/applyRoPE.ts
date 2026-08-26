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

export function argmax(values: Float32Array): number {
  if (values.length === 0) {
    throw new Error("argmax requires at least one value");
  }

  let bestIndex = 0;
  let bestValue = values[0] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }

  return bestIndex;
}

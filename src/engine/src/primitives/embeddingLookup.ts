import type { TensorView } from "../loader";

export function embeddingLookup(
  embedding: TensorView,
  tokenId: number,
  output: Float32Array,
  outputOffset = 0,
): void {
  const [entryCount, embeddingSize] = requireMatrixShape(embedding, "embedding");
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= entryCount) {
    throw new RangeError(`tokenId must be an integer in [0, ${entryCount}), got ${tokenId}`);
  }
  if (!Number.isInteger(outputOffset) || outputOffset < 0) {
    throw new RangeError(`outputOffset must be a non-negative integer, got ${outputOffset}`);
  }
  if (outputOffset + embeddingSize > output.length) {
    throw new RangeError(
      `output is too small: need ${outputOffset + embeddingSize} values, got ${output.length}`,
    );
  }

  const sourceOffset = tokenId * embeddingSize;
  output.set(embedding.data.subarray(sourceOffset, sourceOffset + embeddingSize), outputOffset);
}

function requireMatrixShape(tensor: TensorView, name: string): [number, number] {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }

  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows <= 0 || columns <= 0) {
    throw new Error(`${name} dimensions must be positive, got [${rows}, ${columns}]`);
  }
  if (tensor.data.length !== rows * columns) {
    throw new Error(`${name} data length does not match shape [${rows}, ${columns}]`);
  }

  return [rows, columns];
}

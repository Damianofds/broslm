import type { TensorView } from "../tensor";
import {
  createStorageBuffer,
  destroyBuffers,
  readFloat32Buffer,
  runComputeShader,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

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

export async function embeddingLookupGpu(
  runtime: WebGpuRuntime,
  embedding: TensorView,
  tokenId: number,
): Promise<Float32Array> {
  const [entryCount, embeddingSize] = requireMatrixShape(embedding, "embedding");
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= entryCount) {
    throw new RangeError(`tokenId must be an integer in [0, ${entryCount}), got ${tokenId}`);
  }

  const embeddingBuffer = createStorageBuffer(runtime, embedding.data);
  const outputBuffer = createStorageBuffer(runtime, embeddingSize * Float32Array.BYTES_PER_ELEMENT);
  const paramsBuffer = createStorageBuffer(
    runtime,
    new Uint32Array([tokenId, embeddingSize, 0, 0]),
    webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
  );

  try {
    await runComputeShader(
      runtime,
      embeddingLookupShader,
      [
        { binding: 0, resource: { buffer: embeddingBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
      [Math.ceil(embeddingSize / 128)],
    );
    return readFloat32Buffer(runtime, outputBuffer, embeddingSize);
  } finally {
    destroyBuffers(embeddingBuffer, outputBuffer, paramsBuffer);
  }
}

const embeddingLookupShader = `
struct Params {
  tokenId: u32,
  embeddingSize: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> embedding: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.embeddingSize) {
    return;
  }
  output[index] = embedding[params.tokenId * params.embeddingSize + index];
}
`;

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

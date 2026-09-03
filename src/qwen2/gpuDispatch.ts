export interface QuantizedGemvDispatchPlan {
  shader: "cooperative" | "scalar";
  workgroups: readonly [number, number?];
  rowsPerDispatch: number;
}

const scalarGemvWorkgroupSize = 64;

export function planQuantizedGemvDispatch(
  outputSize: number,
  maxComputeWorkgroupsPerDimension: number,
): QuantizedGemvDispatchPlan {
  if (!Number.isInteger(outputSize) || outputSize <= 0) {
    throw new RangeError(`quantized GEMV output size must be a positive integer, got ${outputSize}`);
  }
  if (
    !Number.isInteger(maxComputeWorkgroupsPerDimension) ||
    maxComputeWorkgroupsPerDimension <= 0
  ) {
    throw new RangeError(
      `maxComputeWorkgroupsPerDimension must be a positive integer, got ` +
        `${maxComputeWorkgroupsPerDimension}`,
    );
  }

  const rowsPerDispatch = Math.min(outputSize, maxComputeWorkgroupsPerDimension);
  const rowDispatches = Math.ceil(outputSize / rowsPerDispatch);
  if (rowDispatches <= maxComputeWorkgroupsPerDimension) {
    return {
      shader: "cooperative",
      workgroups: [rowsPerDispatch, rowDispatches],
      rowsPerDispatch,
    };
  }

  const scalarWorkgroups = Math.ceil(outputSize / scalarGemvWorkgroupSize);
  if (scalarWorkgroups <= maxComputeWorkgroupsPerDimension) {
    return {
      shader: "scalar",
      workgroups: [scalarWorkgroups],
      rowsPerDispatch,
    };
  }

  throw new RangeError(
    `quantized GEMV output size ${outputSize} cannot fit within the WebGPU dispatch limit ` +
      `${maxComputeWorkgroupsPerDimension}: requested cooperative grid ` +
      `[${rowsPerDispatch}, ${rowDispatches}] and scalar grid [${scalarWorkgroups}, 1]`,
  );
}

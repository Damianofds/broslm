export interface QuantizedMatrixDispatchPlan {
  workgroups: readonly [number];
}

const quantizedMatrixWorkgroupSize = 64;

export function planQuantizedMatrixDispatch(
  outputSize: number,
  sequenceLength: number,
  maxComputeWorkgroupsPerDimension: number,
): QuantizedMatrixDispatchPlan {
  for (const [name, value] of [
    ["outputSize", outputSize],
    ["sequenceLength", sequenceLength],
    ["maxComputeWorkgroupsPerDimension", maxComputeWorkgroupsPerDimension],
  ] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive safe integer, got ${value}`);
    }
  }

  const outputElements = outputSize * sequenceLength;
  if (!Number.isSafeInteger(outputElements)) {
    throw new RangeError(
      `quantized matrix output element count is not a safe integer: ` +
        `${outputSize} * ${sequenceLength}`,
    );
  }
  const workgroups = Math.ceil(outputElements / quantizedMatrixWorkgroupSize);
  if (workgroups > maxComputeWorkgroupsPerDimension) {
    throw new RangeError(
      `quantized matrix dispatch [${workgroups}, 1, 1] exceeds ` +
        `maxComputeWorkgroupsPerDimension (${maxComputeWorkgroupsPerDimension})`,
    );
  }
  return { workgroups: [workgroups] };
}

export function shouldUseFusedQkvProjection(options: {
  sequenceLength: number;
  qOutputBaseOffset?: number;
  kOutputBaseOffset?: number;
  vOutputBaseOffset?: number;
  weightsCompatible: boolean;
}): boolean {
  return (
    options.sequenceLength > 0 &&
    (options.qOutputBaseOffset ?? 0) === 0 &&
    (options.kOutputBaseOffset ?? 0) === 0 &&
    (options.vOutputBaseOffset ?? 0) === 0 &&
    options.weightsCompatible
  );
}

export function shouldUseSplitAttentionDecode(position: number, tileSize: number): boolean {
  if (!Number.isSafeInteger(position) || position < 0) {
    throw new RangeError(`position must be a non-negative safe integer, got ${position}`);
  }
  if (!Number.isSafeInteger(tileSize) || tileSize <= 0) {
    throw new RangeError(`tileSize must be a positive safe integer, got ${tileSize}`);
  }
  return position + 1 > tileSize;
}

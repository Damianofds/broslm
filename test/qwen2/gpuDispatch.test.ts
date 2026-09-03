import { describe, expect, it } from "vitest";
import {
  planQuantizedMatrixDispatch,
  shouldUseFusedQkvProjection,
  shouldUseSplitAttentionDecode,
} from "../../src/qwen2/gpuDispatch";

describe("Qwen GPU dispatch policy", () => {
  const deviceLimit = 65_535;

  it("uses a compact scalar dispatch for the production LM head", () => {
    expect(planQuantizedMatrixDispatch(151_936, 1, deviceLimit)).toEqual({
      workgroups: [2_374],
    });
  });

  it("keeps the largest supported prefill projection within the device limit", () => {
    expect(planQuantizedMatrixDispatch(4_864, 256, deviceLimit)).toEqual({
      workgroups: [19_456],
    });
  });

  it("rejects an oversized scalar dispatch before WebGPU encoding", () => {
    expect(() => planQuantizedMatrixDispatch(4_864, 1_024, deviceLimit)).toThrow(
      /\[77824, 1, 1\].*65535/,
    );
  });

  it("fuses compatible QKV projections for prefill and decode", () => {
    expect(
      shouldUseFusedQkvProjection({ sequenceLength: 32, weightsCompatible: true }),
    ).toBe(true);
    expect(
      shouldUseFusedQkvProjection({ sequenceLength: 1, weightsCompatible: true }),
    ).toBe(true);
  });

  it("keeps offset and incompatible projections separate", () => {
    expect(
      shouldUseFusedQkvProjection({ sequenceLength: 32, weightsCompatible: false }),
    ).toBe(false);
    expect(
      shouldUseFusedQkvProjection({
        sequenceLength: 32,
        kOutputBaseOffset: 128,
        weightsCompatible: true,
      }),
    ).toBe(false);
  });

  it("uses direct attention for one tile and split attention above it", () => {
    expect(shouldUseSplitAttentionDecode(0, 256)).toBe(false);
    expect(shouldUseSplitAttentionDecode(255, 256)).toBe(false);
    expect(shouldUseSplitAttentionDecode(256, 256)).toBe(true);
  });
});

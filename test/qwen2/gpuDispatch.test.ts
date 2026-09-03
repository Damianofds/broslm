import { describe, expect, it } from "vitest";
import { planQuantizedGemvDispatch } from "../../src/qwen2/gpuDispatch";

describe("quantized GEMV dispatch planning", () => {
  const deviceLimit = 65_535;

  it("splits the production vocabulary across two legal dimensions", () => {
    expect(planQuantizedGemvDispatch(151_936, deviceLimit)).toEqual({
      shader: "cooperative",
      workgroups: [65_535, 3],
      rowsPerDispatch: 65_535,
    });
  });

  it.each([
    [deviceLimit - 1, [deviceLimit - 1, 1]],
    [deviceLimit, [deviceLimit, 1]],
    [deviceLimit + 1, [deviceLimit, 2]],
  ])("plans output size %i within the device limit", (outputSize, workgroups) => {
    const plan = planQuantizedGemvDispatch(outputSize, deviceLimit);

    expect(plan.shader).toBe("cooperative");
    expect(plan.workgroups).toEqual(workgroups);
    expect(
      plan.workgroups.every(
        (dimension) => dimension !== undefined && dimension <= deviceLimit,
      ),
    ).toBe(true);
  });

  it("uses the scalar kernel only when the cooperative grid cannot fit", () => {
    expect(planQuantizedGemvDispatch(5, 2)).toEqual({
      shader: "scalar",
      workgroups: [1],
      rowsPerDispatch: 2,
    });
  });

  it("rejects an output that neither dispatch layout can fit", () => {
    expect(() => planQuantizedGemvDispatch(129, 2)).toThrow(
      /output size 129.*limit 2.*\[2, 65\].*\[3, 1\]/,
    );
  });
});

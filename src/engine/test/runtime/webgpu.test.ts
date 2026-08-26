import { describe, expect, it, vi } from "vitest";
import {
  createWebGpuRuntime,
  detectWebGpuSupport,
  isWebGpuApiAvailable,
  resolveInferenceBackend,
} from "../../src/runtime/webgpu";

describe("WebGPU runtime detection", () => {
  it("reports missing API", async () => {
    const support = await detectWebGpuSupport({} as Navigator);

    expect(isWebGpuApiAvailable({} as Navigator)).toBe(false);
    expect(support).toMatchObject({
      supported: false,
      apiAvailable: false,
      adapterAvailable: false,
    });
  });

  it("reports missing adapter", async () => {
    const navigatorWithGpu = {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue(null),
      },
    } as unknown as Navigator;

    const support = await detectWebGpuSupport(navigatorWithGpu);

    expect(support).toMatchObject({
      supported: false,
      apiAvailable: true,
      adapterAvailable: false,
    });
  });

  it("reports device request failure", async () => {
    const navigatorWithGpu = {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue({
          requestDevice: vi.fn().mockRejectedValue(new Error("device denied")),
        }),
      },
    } as unknown as Navigator;

    const support = await detectWebGpuSupport(navigatorWithGpu);

    expect(support).toMatchObject({
      supported: false,
      apiAvailable: true,
      adapterAvailable: true,
      reason: "device denied",
    });
  });

  it("reports supported WebGPU", async () => {
    const destroy = vi.fn();
    const navigatorWithGpu = {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue({
          limits: {
            maxBufferSize: 268_435_456,
            maxStorageBufferBindingSize: 2_147_483_644,
          },
          requestDevice: vi.fn().mockResolvedValue({ destroy }),
        }),
      },
    } as unknown as Navigator;

    const support = await detectWebGpuSupport(navigatorWithGpu);

    expect(support).toEqual({
      supported: true,
      apiAvailable: true,
      adapterAvailable: true,
      limits: {
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 2_147_483_644,
      },
    });
    expect(destroy).toHaveBeenCalledOnce();
  });

  it("requests a higher storage buffer binding limit for runtime devices", async () => {
    const requestDevice = vi.fn().mockResolvedValue({ destroy: vi.fn() });
    const navigatorWithGpu = {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue({
          limits: {
            maxBufferSize: 268_435_456,
            maxStorageBufferBindingSize: 2_147_483_644,
          },
          requestDevice,
        }),
      },
    } as unknown as Navigator;

    await createWebGpuRuntime(navigatorWithGpu, {
      requiredStorageBufferBindingSize: 144_643_072,
    });

    expect(requestDevice).toHaveBeenCalledWith({
      requiredLimits: {
        maxStorageBufferBindingSize: 144_643_072,
      },
    });
  });

  it("rejects runtime creation when the adapter storage binding limit is too low", async () => {
    const navigatorWithGpu = {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue({
          limits: {
            maxBufferSize: 268_435_456,
            maxStorageBufferBindingSize: 134_217_728,
          },
          requestDevice: vi.fn(),
        }),
      },
    } as unknown as Navigator;

    await expect(
      createWebGpuRuntime(navigatorWithGpu, {
        requiredStorageBufferBindingSize: 144_643_072,
      }),
    ).rejects.toThrow("maxStorageBufferBindingSize");
  });
});

describe("resolveInferenceBackend", () => {
  it("resolves auto to WebGPU when available", () => {
    expect(
      resolveInferenceBackend({
        preference: "auto",
        webgpuAvailable: true,
      }),
    ).toBe("webgpu");
  });

  it("resolves auto to CPU when WebGPU is unavailable and optional", () => {
    expect(
      resolveInferenceBackend({
        preference: "auto",
        webgpuAvailable: false,
      }),
    ).toBe("cpu");
  });

  it("rejects required WebGPU when unavailable", () => {
    expect(() =>
      resolveInferenceBackend({
        preference: "auto",
        webgpuAvailable: false,
        webgpuRequired: true,
      }),
    ).toThrow("This model requires WebGPU");
  });

  it("honors explicit CPU", () => {
    expect(
      resolveInferenceBackend({
        preference: "cpu",
        webgpuAvailable: true,
        webgpuRequired: true,
      }),
    ).toBe("cpu");
  });

  it("rejects explicit WebGPU when unavailable", () => {
    expect(() =>
      resolveInferenceBackend({
        preference: "webgpu",
        webgpuAvailable: false,
      }),
    ).toThrow("WebGPU backend was requested");
  });
});

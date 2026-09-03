import { describe, expect, it, vi } from "vitest";
import {
  createWebGpuRuntime,
  createStaticStorageBuffer,
  detectWebGpuSupport,
  isWebGpuApiAvailable,
  runComputeShader,
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

  it("enables shader-f16 when the adapter supports it", async () => {
    const requestDevice = vi.fn().mockResolvedValue({ destroy: vi.fn() });
    const navigatorWithGpu = {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue({
          features: new Set(["shader-f16"]),
          limits: {
            maxBufferSize: 268_435_456,
            maxStorageBufferBindingSize: 2_147_483_644,
          },
          requestDevice,
        }),
      },
    } as unknown as Navigator;

    const runtime = await createWebGpuRuntime(navigatorWithGpu);

    expect(requestDevice).toHaveBeenCalledWith({ requiredFeatures: ["shader-f16"] });
    expect(runtime.shaderF16).toBe(true);
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

describe("static GPU buffer cache", () => {
  it("reuses buffers for the same source object", () => {
    const buffer = {};
    const createBuffer = vi.fn().mockReturnValue(buffer);
    const writeBuffer = vi.fn();
    const runtime = {
      backend: "webgpu",
      adapter: {},
      device: {
        createBuffer,
        queue: {
          writeBuffer,
        },
      },
      staticBufferCache: new WeakMap(),
    } as unknown as Parameters<typeof createStaticStorageBuffer>[0];
    const data = new Float32Array([1, 2, 3]);

    const first = createStaticStorageBuffer(runtime, data);
    const second = createStaticStorageBuffer(runtime, data);

    expect(first).toBe(buffer);
    expect(second).toBe(buffer);
    expect(createBuffer).toHaveBeenCalledOnce();
    expect(writeBuffer).toHaveBeenCalledOnce();
  });
});

describe("compute pipeline cache", () => {
  it("reuses compute pipelines for the same shader", async () => {
    const pipeline = {
      getBindGroupLayout: vi.fn().mockReturnValue({}),
    };
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
      end: vi.fn(),
    };
    const commandBuffer = {};
    const createShaderModule = vi.fn().mockReturnValue({});
    const createComputePipeline = vi.fn().mockReturnValue(pipeline);
    const runtime = {
      backend: "webgpu",
      adapter: {},
      device: {
        createShaderModule,
        createComputePipeline,
        createBindGroup: vi.fn().mockReturnValue({}),
        createCommandEncoder: vi.fn().mockReturnValue({
          beginComputePass: vi.fn().mockReturnValue(pass),
          finish: vi.fn().mockReturnValue(commandBuffer),
        }),
        queue: {
          submit: vi.fn(),
          onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined),
        },
      },
      staticBufferCache: new WeakMap(),
      computePipelineCache: new Map(),
    } as unknown as Parameters<typeof runComputeShader>[0];
    const shader = "@compute @workgroup_size(1) fn main() {}";

    await runComputeShader(runtime, shader, [], [1]);
    await runComputeShader(runtime, shader, [], [1]);

    expect(createShaderModule).toHaveBeenCalledOnce();
    expect(createComputePipeline).toHaveBeenCalledOnce();
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(2);
  });
});

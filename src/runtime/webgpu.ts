export type InferenceBackend = "cpu" | "webgpu";
export type InferenceBackendPreference = "auto" | InferenceBackend;

export interface WebGpuSupportStatus {
  supported: boolean;
  apiAvailable: boolean;
  adapterAvailable: boolean;
  limits?: WebGpuLimitSummary;
  reason?: string;
}

export interface WebGpuLimitSummary {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}

export interface WebGpuRuntime {
  backend: "webgpu";
  adapter: GPUAdapter;
  device: GPUDevice;
  staticBufferCache: WeakMap<object, GPUBuffer>;
  computePipelineCache: Map<string, GPUComputePipeline>;
}

export interface WebGpuRuntimeOptions {
  requiredStorageBufferBindingSize?: number;
}

export type WebGpuNavigator = Pick<Navigator, "gpu">;

export interface BackendResolutionOptions {
  preference?: InferenceBackendPreference;
  webgpuRequired?: boolean;
  webgpuAvailable: boolean;
}

export const webGpuBufferUsage = {
  mapRead: 1,
  copySrc: 4,
  copyDst: 8,
  uniform: 64,
  storage: 128,
} as const;

export const webGpuMapMode = {
  read: 1,
} as const;

export function isWebGpuApiAvailable(target: WebGpuNavigator | undefined = globalNavigator()): boolean {
  return Boolean(target && "gpu" in target && target.gpu);
}

export async function detectWebGpuSupport(
  target: WebGpuNavigator | undefined = globalNavigator(),
): Promise<WebGpuSupportStatus> {
  if (!isWebGpuApiAvailable(target)) {
    return {
      supported: false,
      apiAvailable: false,
      adapterAvailable: false,
      reason: "WebGPU API is not available in this environment.",
    };
  }

  try {
    const gpu = target?.gpu;
    if (!gpu) {
      throw new Error("WebGPU API is not available in this environment.");
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return {
        supported: false,
        apiAvailable: true,
        adapterAvailable: false,
        reason: "WebGPU API is available, but no compatible GPU adapter was found.",
      };
    }

    const device = await adapter.requestDevice();
    device.destroy();
    return {
      supported: true,
      apiAvailable: true,
      adapterAvailable: true,
      limits: summarizeAdapterLimits(adapter),
    };
  } catch (error: unknown) {
    return {
      supported: false,
      apiAvailable: true,
      adapterAvailable: true,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function createWebGpuRuntime(
  target: WebGpuNavigator | undefined = globalNavigator(),
  options: WebGpuRuntimeOptions = {},
): Promise<WebGpuRuntime> {
  if (!isWebGpuApiAvailable(target)) {
    throw new Error("WebGPU API is not available in this environment.");
  }

  const gpu = target?.gpu;
  if (!gpu) {
    throw new Error("WebGPU API is not available in this environment.");
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU API is available, but no compatible GPU adapter was found.");
  }

  const device = await requestRuntimeDevice(adapter, options);
  return {
    backend: "webgpu",
    adapter,
    device,
    staticBufferCache: new WeakMap(),
    computePipelineCache: new Map(),
  };
}

function requestRuntimeDevice(
  adapter: GPUAdapter,
  options: WebGpuRuntimeOptions,
): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  const requiredStorageBufferBindingSize = options.requiredStorageBufferBindingSize;

  if (requiredStorageBufferBindingSize !== undefined) {
    if (
      !Number.isFinite(requiredStorageBufferBindingSize) ||
      requiredStorageBufferBindingSize <= 0
    ) {
      throw new RangeError(
        `requiredStorageBufferBindingSize must be positive, got ${requiredStorageBufferBindingSize}`,
      );
    }
    if (requiredStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
      throw new Error(
        `This WebGPU adapter supports maxStorageBufferBindingSize ` +
          `${adapter.limits.maxStorageBufferBindingSize}, but this model needs ` +
          `${requiredStorageBufferBindingSize}.`,
      );
    }

    requiredLimits.maxStorageBufferBindingSize = Math.ceil(requiredStorageBufferBindingSize);
  }

  return adapter.requestDevice(
    Object.keys(requiredLimits).length > 0 ? { requiredLimits } : undefined,
  );
}

function summarizeAdapterLimits(adapter: GPUAdapter): WebGpuLimitSummary {
  return {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

export function resolveInferenceBackend(options: BackendResolutionOptions): InferenceBackend {
  const preference = options.preference ?? "auto";
  if (preference === "cpu") {
    return "cpu";
  }
  if (preference === "webgpu") {
    if (!options.webgpuAvailable) {
      throw new Error("WebGPU backend was requested, but WebGPU is not available.");
    }
    return "webgpu";
  }
  if (options.webgpuAvailable) {
    return "webgpu";
  }
  if (options.webgpuRequired) {
    throw new Error("This model requires WebGPU, but WebGPU is not available.");
  }
  return "cpu";
}

export function createStorageBuffer(
  runtime: WebGpuRuntime,
  dataOrByteLength: Float32Array | Uint32Array | Uint8Array | ArrayBuffer | number,
  usage: GPUBufferUsageFlags =
    webGpuBufferUsage.storage | webGpuBufferUsage.copyDst | webGpuBufferUsage.copySrc,
): GPUBuffer {
  const byteLength = typeof dataOrByteLength === "number" ? dataOrByteLength : dataOrByteLength.byteLength;
  const buffer = runtime.device.createBuffer({
    size: alignedByteLength(byteLength),
    usage,
  });
  if (typeof dataOrByteLength !== "number") {
    const data =
      dataOrByteLength instanceof ArrayBuffer
        ? dataOrByteLength
        : dataOrByteLength.buffer.slice(
            dataOrByteLength.byteOffset,
            dataOrByteLength.byteOffset + dataOrByteLength.byteLength,
          );
    runtime.device.queue.writeBuffer(buffer, 0, data);
  }
  return buffer;
}

export function createStaticStorageBuffer(
  runtime: WebGpuRuntime,
  data: Float32Array | Uint32Array | Uint8Array | ArrayBuffer,
  usage: GPUBufferUsageFlags =
    webGpuBufferUsage.storage | webGpuBufferUsage.copyDst | webGpuBufferUsage.copySrc,
): GPUBuffer {
  const cached = runtime.staticBufferCache.get(data);
  if (cached) {
    return cached;
  }

  const buffer = createStorageBuffer(runtime, data, usage);
  runtime.staticBufferCache.set(data, buffer);
  return buffer;
}

export async function readFloat32Buffer(
  runtime: WebGpuRuntime,
  source: GPUBuffer,
  valueCount: number,
): Promise<Float32Array> {
  const byteLength = valueCount * Float32Array.BYTES_PER_ELEMENT;
  const readBuffer = runtime.device.createBuffer({
    size: alignedByteLength(byteLength),
    usage: webGpuBufferUsage.copyDst | webGpuBufferUsage.mapRead,
  });
  const encoder = runtime.device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteLength);
  runtime.device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(webGpuMapMode.read);
  const mapped = readBuffer.getMappedRange(0, byteLength);
  const copy = new Float32Array(valueCount);
  copy.set(new Float32Array(mapped));
  readBuffer.unmap();
  readBuffer.destroy();
  return copy;
}

export async function runComputeShader(
  runtime: WebGpuRuntime,
  shaderCode: string,
  entries: GPUBindGroupEntry[],
  workgroups: readonly [number, number?, number?],
  constants?: Record<string, number>,
): Promise<void> {
  const encoder = runtime.device.createCommandEncoder();
  encodeComputeShader(runtime, encoder, shaderCode, entries, workgroups, constants);
  runtime.device.queue.submit([encoder.finish()]);
}

export function encodeComputeShader(
  runtime: WebGpuRuntime,
  encoder: GPUCommandEncoder,
  shaderCode: string,
  entries: GPUBindGroupEntry[],
  workgroups: readonly [number, number?, number?],
  constants?: Record<string, number>,
): void {
  const pipeline = getCachedComputePipeline(runtime, shaderCode, constants);
  const bindGroup = runtime.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
  pass.end();
}

function getCachedComputePipeline(
  runtime: WebGpuRuntime,
  shaderCode: string,
  constants?: Record<string, number>,
): GPUComputePipeline {
  const cacheKey = computePipelineCacheKey(shaderCode, constants);
  const cached = runtime.computePipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const shaderModule = runtime.device.createShaderModule({ code: shaderCode });
  const pipeline = runtime.device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main",
      constants,
    },
  });
  runtime.computePipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

function computePipelineCacheKey(
  shaderCode: string,
  constants?: Record<string, number>,
): string {
  if (!constants) {
    return shaderCode;
  }

  const constantKey = Object.keys(constants)
    .sort()
    .map((key) => `${key}:${constants[key]}`)
    .join(",");
  return `${shaderCode}\nconstants:${constantKey}`;
}

export function destroyBuffers(...buffers: Array<GPUBuffer | undefined>): void {
  for (const buffer of buffers) {
    buffer?.destroy();
  }
}

export function destroyWebGpuRuntime(runtime: WebGpuRuntime | undefined): void {
  if (!runtime) {
    return;
  }
  runtime.staticBufferCache = new WeakMap();
  runtime.computePipelineCache.clear();
  runtime.device.destroy();
}

function alignedByteLength(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function globalNavigator(): WebGpuNavigator | undefined {
  return typeof navigator === "undefined" ? undefined : navigator;
}

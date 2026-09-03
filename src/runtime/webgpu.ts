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
  bindGroupCache: Map<string, GPUBindGroup>;
  resourceIds: WeakMap<object, number>;
  nextResourceId: number;
  diagnostics: WebGpuRuntimeDiagnostics;
  shaderF16: boolean;
}

export interface WebGpuRuntimeDiagnostics {
  buffersCreated: number;
  buffersDestroyed: number;
  bytesAllocated: number;
  currentBytesAllocated: number;
  peakBytesAllocated: number;
  bindGroupsCreated: number;
  computePassesEncoded: number;
  commandSubmissions: number;
  bytesReadBack: number;
}

export interface WebGpuRuntimeOptions {
  requiredStorageBufferBindingSize?: number;
}

export type WebGpuNavigator = Pick<Navigator, "gpu">;

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

const bufferAllocations = new WeakMap<
  GPUBuffer,
  { runtime: WebGpuRuntime; byteLength: number; destroyed: boolean }
>();

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
    bindGroupCache: new Map(),
    resourceIds: new WeakMap(),
    nextResourceId: 1,
    diagnostics: createEmptyWebGpuRuntimeDiagnostics(),
    shaderF16: adapter.features?.has?.("shader-f16") ?? false,
  };
}

function requestRuntimeDevice(
  adapter: GPUAdapter,
  options: WebGpuRuntimeOptions,
): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  const requiredFeatures: GPUFeatureName[] = adapter.features?.has?.("shader-f16")
    ? ["shader-f16"]
    : [];
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

  const descriptor: GPUDeviceDescriptor = {};
  if (Object.keys(requiredLimits).length > 0) descriptor.requiredLimits = requiredLimits;
  if (requiredFeatures.length > 0) descriptor.requiredFeatures = requiredFeatures;
  return adapter.requestDevice(
    Object.keys(descriptor).length > 0 ? descriptor : undefined,
  );
}

function summarizeAdapterLimits(adapter: GPUAdapter): WebGpuLimitSummary {
  return {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

export function createStorageBuffer(
  runtime: WebGpuRuntime,
  dataOrByteLength: Float32Array | Uint32Array | Uint8Array | ArrayBuffer | number,
  usage: GPUBufferUsageFlags =
    webGpuBufferUsage.storage | webGpuBufferUsage.copyDst | webGpuBufferUsage.copySrc,
): GPUBuffer {
  ensureWebGpuRuntimeInstrumentation(runtime);
  const byteLength = typeof dataOrByteLength === "number" ? dataOrByteLength : dataOrByteLength.byteLength;
  const buffer = runtime.device.createBuffer({
    size: alignedByteLength(byteLength),
    usage,
  });
  trackBufferCreation(runtime, buffer, alignedByteLength(byteLength));
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
  ensureWebGpuRuntimeInstrumentation(runtime);
  const byteLength = valueCount * Float32Array.BYTES_PER_ELEMENT;
  const readBuffer = runtime.device.createBuffer({
    size: alignedByteLength(byteLength),
    usage: webGpuBufferUsage.copyDst | webGpuBufferUsage.mapRead,
  });
  trackBufferCreation(runtime, readBuffer, alignedByteLength(byteLength));
  const encoder = runtime.device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteLength);
  submitWebGpuCommands(runtime, [encoder.finish()]);
  await readBuffer.mapAsync(webGpuMapMode.read);
  const mapped = readBuffer.getMappedRange(0, byteLength);
  const copy = new Float32Array(valueCount);
  copy.set(new Float32Array(mapped));
  readBuffer.unmap();
  destroyTrackedBuffer(readBuffer);
  runtime.diagnostics.bytesReadBack += byteLength;
  return copy;
}

export async function readUint32Buffer(
  runtime: WebGpuRuntime,
  source: GPUBuffer,
  valueCount: number,
): Promise<Uint32Array> {
  ensureWebGpuRuntimeInstrumentation(runtime);
  const byteLength = valueCount * Uint32Array.BYTES_PER_ELEMENT;
  const readBuffer = runtime.device.createBuffer({
    size: alignedByteLength(byteLength),
    usage: webGpuBufferUsage.copyDst | webGpuBufferUsage.mapRead,
  });
  trackBufferCreation(runtime, readBuffer, alignedByteLength(byteLength));
  const encoder = runtime.device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteLength);
  submitWebGpuCommands(runtime, [encoder.finish()]);
  await readBuffer.mapAsync(webGpuMapMode.read);
  const mapped = readBuffer.getMappedRange(0, byteLength);
  const copy = new Uint32Array(valueCount);
  copy.set(new Uint32Array(mapped));
  readBuffer.unmap();
  destroyTrackedBuffer(readBuffer);
  runtime.diagnostics.bytesReadBack += byteLength;
  return copy;
}

export interface WebGpuBufferCopy {
  source: GPUBuffer;
  sourceOffset?: number;
  destinationOffset: number;
  byteLength: number;
}

export async function readGpuBufferCopies(
  runtime: WebGpuRuntime,
  copies: readonly WebGpuBufferCopy[],
  destination: GPUBuffer,
  byteLength: number,
): Promise<ArrayBuffer> {
  ensureWebGpuRuntimeInstrumentation(runtime);
  const encoder = runtime.device.createCommandEncoder();
  for (const copy of copies) {
    encoder.copyBufferToBuffer(
      copy.source,
      copy.sourceOffset ?? 0,
      destination,
      copy.destinationOffset,
      copy.byteLength,
    );
  }
  submitWebGpuCommands(runtime, [encoder.finish()]);
  await destination.mapAsync(webGpuMapMode.read, 0, byteLength);
  const result = destination.getMappedRange(0, byteLength).slice(0);
  destination.unmap();
  runtime.diagnostics.bytesReadBack += byteLength;
  return result;
}

export async function readMappedGpuBuffer(
  runtime: WebGpuRuntime,
  source: GPUBuffer,
  byteLength: number,
): Promise<ArrayBuffer> {
  ensureWebGpuRuntimeInstrumentation(runtime);
  await source.mapAsync(webGpuMapMode.read, 0, byteLength);
  const result = source.getMappedRange(0, byteLength).slice(0);
  source.unmap();
  runtime.diagnostics.bytesReadBack += byteLength;
  return result;
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
  submitWebGpuCommands(runtime, [encoder.finish()]);
}

export function encodeComputeShader(
  runtime: WebGpuRuntime,
  encoder: GPUCommandEncoder,
  shaderCode: string,
  entries: GPUBindGroupEntry[],
  workgroups: readonly [number, number?, number?],
  constants?: Record<string, number>,
): void {
  ensureWebGpuRuntimeInstrumentation(runtime);
  const pipeline = getCachedComputePipeline(runtime, shaderCode, constants);
  const bindGroup = getCachedBindGroup(runtime, pipeline, entries);
  const pass = encoder.beginComputePass();
  runtime.diagnostics.computePassesEncoded += 1;
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
  pass.end();
}

function getCachedBindGroup(
  runtime: WebGpuRuntime,
  pipeline: GPUComputePipeline,
  entries: GPUBindGroupEntry[],
): GPUBindGroup {
  const key = `${resourceId(runtime, pipeline)}:${entries.map((entry) => bindGroupEntryKey(runtime, entry)).join("|")}`;
  const cached = runtime.bindGroupCache.get(key);
  if (cached) {
    return cached;
  }
  const bindGroup = runtime.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  runtime.bindGroupCache.set(key, bindGroup);
  runtime.diagnostics.bindGroupsCreated += 1;
  return bindGroup;
}

function bindGroupEntryKey(runtime: WebGpuRuntime, entry: GPUBindGroupEntry): string {
  const resource = entry.resource as GPUBufferBinding;
  if (resource && typeof resource === "object" && "buffer" in resource) {
    return `${entry.binding}:b${resourceId(runtime, resource.buffer)}:${resource.offset ?? 0}:${resource.size ?? 0}`;
  }
  return `${entry.binding}:r${resourceId(runtime, entry.resource as object)}`;
}

function resourceId(runtime: WebGpuRuntime, resource: object): number {
  const existing = runtime.resourceIds.get(resource);
  if (existing !== undefined) {
    return existing;
  }
  const id = runtime.nextResourceId;
  runtime.nextResourceId += 1;
  runtime.resourceIds.set(resource, id);
  return id;
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
    if (buffer) destroyTrackedBuffer(buffer);
  }
}

export function submitWebGpuCommands(
  runtime: WebGpuRuntime,
  commandBuffers: readonly GPUCommandBuffer[],
): void {
  ensureWebGpuRuntimeInstrumentation(runtime);
  runtime.device.queue.submit(commandBuffers);
  runtime.diagnostics.commandSubmissions += 1;
}

export function clearWebGpuBindGroupCache(runtime: WebGpuRuntime): void {
  runtime.bindGroupCache.clear();
}

export function snapshotWebGpuRuntimeDiagnostics(
  runtime: WebGpuRuntime,
): Readonly<WebGpuRuntimeDiagnostics> {
  ensureWebGpuRuntimeInstrumentation(runtime);
  return { ...runtime.diagnostics };
}

export function destroyWebGpuRuntime(runtime: WebGpuRuntime | undefined): void {
  if (!runtime) {
    return;
  }
  runtime.staticBufferCache = new WeakMap();
  runtime.computePipelineCache.clear();
  runtime.bindGroupCache.clear();
  runtime.diagnostics.currentBytesAllocated = 0;
  runtime.device.destroy();
}

function createEmptyWebGpuRuntimeDiagnostics(): WebGpuRuntimeDiagnostics {
  return {
    buffersCreated: 0,
    buffersDestroyed: 0,
    bytesAllocated: 0,
    currentBytesAllocated: 0,
    peakBytesAllocated: 0,
    bindGroupsCreated: 0,
    computePassesEncoded: 0,
    commandSubmissions: 0,
    bytesReadBack: 0,
  };
}

function trackBufferCreation(runtime: WebGpuRuntime, buffer: GPUBuffer, byteLength: number): void {
  runtime.diagnostics.buffersCreated += 1;
  runtime.diagnostics.bytesAllocated += byteLength;
  runtime.diagnostics.currentBytesAllocated += byteLength;
  runtime.diagnostics.peakBytesAllocated = Math.max(
    runtime.diagnostics.peakBytesAllocated,
    runtime.diagnostics.currentBytesAllocated,
  );
  bufferAllocations.set(buffer, { runtime, byteLength, destroyed: false });
}

function destroyTrackedBuffer(buffer: GPUBuffer): void {
  const allocation = bufferAllocations.get(buffer);
  if (allocation && !allocation.destroyed) {
    allocation.destroyed = true;
    allocation.runtime.diagnostics.buffersDestroyed += 1;
    allocation.runtime.diagnostics.currentBytesAllocated = Math.max(
      0,
      allocation.runtime.diagnostics.currentBytesAllocated - allocation.byteLength,
    );
  }
  buffer.destroy();
}

function ensureWebGpuRuntimeInstrumentation(runtime: WebGpuRuntime): void {
  runtime.bindGroupCache ??= new Map();
  runtime.resourceIds ??= new WeakMap();
  runtime.nextResourceId ??= 1;
  runtime.diagnostics ??= createEmptyWebGpuRuntimeDiagnostics();
}

function alignedByteLength(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function globalNavigator(): WebGpuNavigator | undefined {
  return typeof navigator === "undefined" ? undefined : navigator;
}

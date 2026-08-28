import {
  allocateQwen2ModelKvCache,
  resetQwen2ModelKvCache,
  type Qwen2ModelKvCache,
} from "./engine/src/qwen2/attentionCache";
import type { GgufTensorInfo } from "./engine/src/qwen2/gguf";
import {
  loadQwen2Model,
  summarizeLoadedQwen2Model,
  type LoadedQwen2Model,
  type Qwen2LoaderProgress,
} from "./engine/src/qwen2/loader";
import { qwen2NextTokenWithCacheBackend } from "./engine/src/qwen2/model";
import {
  qwen2WebGpuCacheSequenceLength,
  qwen2WebGpuPrefillSafetyError,
  qwen2WebGpuSafetyLimits,
} from "./engine/src/qwen2/webgpuSafety";
import {
  createWebGpuRuntime,
  detectWebGpuSupport,
  resolveInferenceBackend,
  type InferenceBackend,
  type WebGpuRuntime,
} from "./engine/src/runtime/webgpu";
import { createQwen2TokenizerPartsFromGgufMetadata } from "./engine/src/qwen2/tokenizer";
import {
  modelCatalog,
  type AppLoadedModelSummary,
  type AppInferencePerformance,
  type AppLoaderProgress,
  type AppTensorSummary,
  type AppWorkerRequest,
  type AppWorkerResponse,
} from "./modelCatalog";
import { createModelCacheFetch } from "./modelCache";

interface ModelWorkerScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<AppWorkerRequest>) => void,
  ): void;
  postMessage(message: AppWorkerResponse): void;
}

type WorkerLoadedModel = {
  modelId: AppWorkerRequest["modelId"];
  model: LoadedQwen2Model;
  cache: Qwen2ModelKvCache;
  backend: InferenceBackend;
  runtime?: WebGpuRuntime;
};

interface WorkerNextTokenResult {
  tokenId: number;
  performance?: AppInferencePerformance;
}

let workerLoadedModel: WorkerLoadedModel | null = null;
let activeQwenInferenceRequestId: string | null = null;

export function installModelWorker(
  selfScope: ModelWorkerScope = globalThis as unknown as ModelWorkerScope,
  fetchImpl?: typeof fetch,
): void {
  selfScope.addEventListener("message", (event: MessageEvent<AppWorkerRequest>) => {
    const message = event.data;
    if (!message) {
      return;
    }

    if (message.type === "next-token") {
      void handleNextTokenMessage(selfScope, message);
      return;
    }

    if (message.type !== "load-model") {
      return;
    }

    workerLoadedModel = null;
    void loadRequestedModel(selfScope, message, fetchImpl);
  });
}

async function handleNextTokenMessage(
  selfScope: ModelWorkerScope,
  message: Extract<AppWorkerRequest, { type: "next-token" }>,
): Promise<void> {
  let ownsActiveQwenRequest = false;
  try {
    if (!workerLoadedModel) {
      throw new Error("Model must be loaded before running next-token inference");
    }
    if (workerLoadedModel.modelId !== message.modelId) {
      throw new Error(
        `Loaded model is ${workerLoadedModel.modelId}, but request expected ${message.modelId}`,
      );
    }

    const result = await runQwen2NextToken(workerLoadedModel, message, () => {
      ownsActiveQwenRequest = true;
    });

    selfScope.postMessage({
      type: "next-token-result",
      requestId: message.requestId,
      modelId: message.modelId,
      tokenId: result.tokenId,
      performance: result.performance,
    });
  } catch (error: unknown) {
    selfScope.postMessage({
      type: "model-error",
      requestId: message.requestId,
      modelId: message.modelId,
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    if (ownsActiveQwenRequest) {
      activeQwenInferenceRequestId = null;
    }
  }
}

async function runQwen2NextToken(
  workerModel: WorkerLoadedModel,
  message: Extract<AppWorkerRequest, { type: "next-token" }>,
  markActive: () => void,
): Promise<WorkerNextTokenResult> {
  if (activeQwenInferenceRequestId) {
    throw new Error("Generation is still running. Wait for the current token to finish.");
  }

  activeQwenInferenceRequestId = message.requestId ?? "qwen-inference";
  markActive();
  if (message.resetCache) {
    resetQwen2ModelKvCache(workerModel.cache);
  }
  validateQwen2WebGpuWorkload(workerModel, message);
  logQwen2WebGpuWorkload(workerModel, message);

  const performanceProfile = qwen2InferencePerformanceProfile(workerModel.cache, message);
  const startedAt = nowMs();
  const result = await qwen2NextTokenWithCacheBackend(
    workerModel.model,
    message.inputIds,
    workerModel.cache,
    {
      backend: workerModel.backend,
      runtime: workerModel.runtime,
      temperature: message.temperature,
      topK: message.topK,
    },
  );
  const elapsedMs = nowMs() - startedAt;

  return {
    tokenId: result.tokenId,
    performance: createInferencePerformance(performanceProfile, elapsedMs),
  };
}

function qwen2InferencePerformanceProfile(
  cache: Qwen2ModelKvCache,
  message: Extract<AppWorkerRequest, { type: "next-token" }>,
): Pick<AppInferencePerformance, "phase" | "tokenCount"> {
  const phase = qwen2RequestNeedsPrefill(cache.inputIds, message.inputIds, message.resetCache)
    ? "prefill"
    : "decode";
  const tokenCount =
    phase === "prefill"
      ? message.inputIds.length
      : Math.max(1, message.inputIds.length - cache.inputIds.length);
  return { phase, tokenCount };
}

function createInferencePerformance(
  performanceProfile: Pick<AppInferencePerformance, "phase" | "tokenCount">,
  elapsedMs: number,
): AppInferencePerformance {
  return {
    phase: performanceProfile.phase,
    tokenCount: performanceProfile.tokenCount,
    elapsedMs,
    tokensPerSecond: tokensPerSecond(performanceProfile.tokenCount, elapsedMs),
  };
}

function tokensPerSecond(tokenCount: number, elapsedMs: number): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    return 0;
  }
  return tokenCount / (elapsedMs / 1000);
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function validateQwen2WebGpuWorkload(
  workerModel: WorkerLoadedModel,
  message: Extract<AppWorkerRequest, { type: "next-token" }>,
): void {
  if (workerModel.backend !== "webgpu") {
    return;
  }

  if (message.inputIds.length > workerModel.cache.maximumSequenceLength) {
    throw new Error(
      `Qwen WebGPU cache is capped at ${workerModel.cache.maximumSequenceLength} tokens ` +
        `for GPU stability. Current request has ${message.inputIds.length} tokens.`,
    );
  }

  if (qwen2RequestNeedsPrefill(workerModel.cache.inputIds, message.inputIds, message.resetCache)) {
    const safetyError = qwen2WebGpuPrefillSafetyError(message.inputIds.length);
    if (safetyError) {
      throw new Error(safetyError);
    }
  }
}

function logQwen2WebGpuWorkload(
  workerModel: WorkerLoadedModel,
  message: Extract<AppWorkerRequest, { type: "next-token" }>,
): void {
  if (workerModel.backend !== "webgpu") {
    return;
  }

  const tokenCount = message.inputIds.length;
  const cacheLength = workerModel.cache.inputIds.length;
  const prefill = qwen2RequestNeedsPrefill(
    workerModel.cache.inputIds,
    message.inputIds,
    message.resetCache,
  );
  console.info("[broslm] Qwen WebGPU workload", {
    requestId: message.requestId,
    mode: prefill ? "prefill" : "decode",
    promptTokens: tokenCount,
    cacheTokens: cacheLength,
    cacheLimit: workerModel.cache.maximumSequenceLength,
    prefillLimit: qwen2WebGpuSafetyLimits.maxPrefillTokens,
    layers: workerModel.model.config.numberOfLayers,
    hiddenSize: workerModel.model.config.hiddenSize,
    keyValueHiddenSize: workerModel.model.config.keyValueHiddenSize,
    estimatedAttentionCells: prefill ? tokenCount * tokenCount * workerModel.model.config.numberOfHeads : tokenCount,
  });
}

function qwen2RequestNeedsPrefill(
  cacheInputIds: readonly number[],
  inputIds: readonly number[],
  resetCache = false,
): boolean {
  if (resetCache || cacheInputIds.length === inputIds.length || cacheInputIds.length > inputIds.length) {
    return true;
  }

  for (let index = 0; index < cacheInputIds.length; index += 1) {
    if (cacheInputIds[index] !== inputIds[index]) {
      return true;
    }
  }
  return false;
}

async function loadRequestedModel(
  selfScope: ModelWorkerScope,
  message: Extract<AppWorkerRequest, { type: "load-model" }>,
  fetchImpl?: typeof fetch,
): Promise<void> {
  try {
    const backend = await resolveRequestedBackend(message);
    const runtime =
      backend === "webgpu"
        ? await createWebGpuRuntime(undefined, {
            requiredStorageBufferBindingSize:
              modelCatalog[message.modelId].backendPolicy.minimumStorageBufferBindingSize,
          })
        : undefined;

    const loadedModel = await loadQwen2Model({
      baseUrl: message.baseUrl,
      ggufPath: message.ggufPath,
      ggufFallbackUrls: message.ggufFallbackUrls,
      fetchImpl,
      onProgress: (progress) => postProgress(selfScope, message, progress),
    });
    const tokenizerParts = createQwen2TokenizerPartsFromGgufMetadata(loadedModel.gguf.metadata);

    workerLoadedModel = {
      modelId: message.modelId,
      model: loadedModel,
      cache: allocateQwen2ModelKvCache(
        loadedModel.config,
        backend === "webgpu"
          ? qwen2WebGpuCacheSequenceLength(loadedModel.config.maximumSequenceLength)
          : loadedModel.config.maximumSequenceLength,
      ),
      backend,
      runtime,
    };
    selfScope.postMessage({
      type: "model-ready",
      requestId: message.requestId,
      modelId: message.modelId,
      backend,
      summary: summarizeQwenForApp(loadedModel, backend, message.modelId),
      tokenizer: {
        kind: "qwen2-gguf",
        parts: tokenizerParts,
      },
    });
  } catch (error: unknown) {
    workerLoadedModel = null;
    selfScope.postMessage({
      type: "model-error",
      requestId: message.requestId,
      modelId: message.modelId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

async function resolveRequestedBackend(
  message: Extract<AppWorkerRequest, { type: "load-model" }>,
): Promise<InferenceBackend> {
  const support = await detectWebGpuSupport();
  return resolveInferenceBackend({
    preference: message.backendPreference,
    webgpuRequired: message.webgpuRequired,
    webgpuAvailable: support.supported,
  });
}

function postProgress(
  selfScope: ModelWorkerScope,
  message: Extract<AppWorkerRequest, { type: "load-model" }>,
  progress: Qwen2LoaderProgress,
): void {
  selfScope.postMessage({
    type: "model-progress",
    requestId: message.requestId,
    modelId: message.modelId,
    progress: {
      ...progress,
      modelId: message.modelId,
    } satisfies AppLoaderProgress,
  });
}

function summarizeQwenForApp(
  model: LoadedQwen2Model,
  backend: InferenceBackend = "cpu",
  modelId?: AppWorkerRequest["modelId"],
): AppLoadedModelSummary {
  const resolvedModelId = modelId ?? "qwen";
  const summary = summarizeLoadedQwen2Model(model);
  return {
    kind: "qwen2",
    modelId: resolvedModelId,
    modelLabel: modelCatalog[resolvedModelId].label,
    backend,
    architecture: summary.architecture,
    dtype: describeGgufDtype(model),
    tensorCount: summary.tensorCount,
    totalByteLength: summary.totalByteLength,
    layers: summary.layers,
    hiddenSize: summary.hiddenSize,
    vocabularySize: summary.vocabularySize,
    maximumSequenceLength: summary.maximumSequenceLength,
    keyValueHiddenSize: model.config.keyValueHiddenSize,
    config: model.config,
    tensors: summarizeQwenTensors(model),
  };
}

function summarizeQwenTensors(model: LoadedQwen2Model): AppTensorSummary[] {
  return [...model.gguf.tensors.values()].map((tensor) => ({
    name: tensor.name,
    description: describeQwenTensor(tensor),
    shape: logicalShape(tensor.dimensions),
    byteOffset: tensor.byteOffset,
    byteLength: tensor.byteLength,
    elementCount: product(tensor.dimensions),
  }));
}

function describeGgufDtype(model: LoadedQwen2Model): string {
  const types = new Set([...model.gguf.tensors.values()].map((tensor) => tensor.type));
  return `GGUF ${[...types].map(ggmlTypeLabel).sort().join("/")}`;
}

function describeQwenTensor(tensor: GgufTensorInfo): string {
  const layerMatch = /^blk\.(\d+)\.(.+)$/.exec(tensor.name);
  if (layerMatch) {
    return `Layer ${layerMatch[1]} ${layerMatch[2]} tensor.`;
  }
  if (tensor.name === "token_embd.weight") {
    return "Token embedding table.";
  }
  if (tensor.name === "output.weight") {
    return "Language-model output projection.";
  }
  if (tensor.name === "output_norm.weight") {
    return "Final RMSNorm scale vector.";
  }
  return "Qwen2 GGUF tensor.";
}

function ggmlTypeLabel(type: number): string {
  switch (type) {
    case 0:
      return "F32";
    case 1:
      return "F16";
    case 2:
      return "Q4_0";
    case 6:
      return "Q5_0";
    case 7:
      return "Q5_1";
    case 8:
      return "Q8_0";
    case 10:
      return "Q2_K";
    case 19:
      return "IQ1_S";
    case 20:
      return "IQ4_NL";
    case 29:
      return "IQ1_M";
    default:
      return `type${type}`;
  }
}

function logicalShape(dimensions: readonly number[]): number[] {
  if (dimensions.length === 2) {
    return [dimensions[1] ?? 0, dimensions[0] ?? 0];
  }
  return [...dimensions];
}

function product(values: readonly number[]): number {
  return values.reduce((total, value) => total * value, 1);
}

installModelWorker(undefined, createModelCacheFetch());

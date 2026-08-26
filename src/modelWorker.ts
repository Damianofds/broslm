import { allocateModelKvCache, type ModelKvCache } from "./engine/src/gpt-neo/attentionCache";
import {
  loadModel as loadGptNeoModel,
  summarizeLoadedModel,
  type LoadedModel as LoadedGptNeoModel,
  type LoaderProgress as GptNeoLoaderProgress,
} from "./engine/src/gpt-neo/loader";
import { nextTokenWithCacheBackend as nextGptNeoTokenWithCacheBackend } from "./engine/src/gpt-neo/model";
import {
  allocateQwen2ModelKvCache,
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
  createWebGpuRuntime,
  detectWebGpuSupport,
  resolveInferenceBackend,
  type InferenceBackend,
  type WebGpuRuntime,
} from "./engine/src/runtime/webgpu";
import { createQwen2TokenizerPartsFromGgufMetadata } from "./engine/src/qwen2/tokenizer";
import {
  modelCatalog,
  normalizeGptNeoSummary,
  type AppLoadedModelSummary,
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

type WorkerLoadedModel =
  | {
      modelId: "tinystories";
      model: LoadedGptNeoModel;
      cache: ModelKvCache;
      backend: InferenceBackend;
      runtime?: WebGpuRuntime;
    }
  | {
      modelId: "qwen";
      model: LoadedQwen2Model;
      cache: Qwen2ModelKvCache;
      backend: InferenceBackend;
      runtime?: WebGpuRuntime;
    };

let workerLoadedModel: WorkerLoadedModel | null = null;

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
  try {
    if (!workerLoadedModel) {
      throw new Error("Model must be loaded before running next-token inference");
    }
    if (workerLoadedModel.modelId !== message.modelId) {
      throw new Error(
        `Loaded model is ${workerLoadedModel.modelId}, but request expected ${message.modelId}`,
      );
    }

    const result =
      workerLoadedModel.modelId === "tinystories"
        ? await nextGptNeoTokenWithCacheBackend(
            workerLoadedModel.model,
            message.inputIds,
            workerLoadedModel.cache,
            {
              backend: workerLoadedModel.backend,
              runtime: workerLoadedModel.runtime,
              temperature: message.temperature,
              topK: message.topK,
            },
          )
        : await qwen2NextTokenWithCacheBackend(
            workerLoadedModel.model,
            message.inputIds,
            workerLoadedModel.cache,
            {
              backend: workerLoadedModel.backend,
              runtime: workerLoadedModel.runtime,
              temperature: message.temperature,
              topK: message.topK,
            },
          );

    selfScope.postMessage({
      type: "next-token-result",
      requestId: message.requestId,
      modelId: message.modelId,
      tokenId: result.tokenId,
    });
  } catch (error: unknown) {
    selfScope.postMessage({
      type: "model-error",
      requestId: message.requestId,
      modelId: message.modelId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
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

    if (message.modelId === "tinystories") {
      const loadedModel = await loadGptNeoModel({
        baseUrl: message.baseUrl,
        configPath: message.configPath,
        weightsIndexPath: message.weightsIndexPath,
        weightsBinaryPath: message.weightsBinaryPath,
        scratchSequenceLength: message.scratchSequenceLength,
        fetchImpl,
        onProgress: (progress) => postProgress(selfScope, message, progress),
      });

      workerLoadedModel = {
        modelId: "tinystories",
        model: loadedModel,
        cache: allocateModelKvCache(loadedModel.config),
        backend,
        runtime,
      };
      selfScope.postMessage({
        type: "model-ready",
        requestId: message.requestId,
        modelId: "tinystories",
        backend,
        summary: normalizeGptNeoSummary(summarizeLoadedModel(loadedModel), backend),
      });
      return;
    }

    const loadedModel = await loadQwen2Model({
      baseUrl: message.baseUrl,
      ggufPath: message.ggufPath,
      ggufFallbackUrls: message.ggufFallbackUrls,
      fetchImpl,
      onProgress: (progress) => postProgress(selfScope, message, progress),
    });
    const tokenizerParts = createQwen2TokenizerPartsFromGgufMetadata(loadedModel.gguf.metadata);

    workerLoadedModel = {
      modelId: "qwen",
      model: loadedModel,
      cache: allocateQwen2ModelKvCache(loadedModel.config),
      backend,
      runtime,
    };
    selfScope.postMessage({
      type: "model-ready",
      requestId: message.requestId,
      modelId: "qwen",
      backend,
      summary: summarizeQwenForApp(loadedModel, backend),
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
  progress: GptNeoLoaderProgress | Qwen2LoaderProgress,
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
): AppLoadedModelSummary {
  const summary = summarizeLoadedQwen2Model(model);
  return {
    kind: "qwen2",
    modelId: "qwen",
    modelLabel: modelCatalog.qwen.label,
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
    case 8:
      return "Q8_0";
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

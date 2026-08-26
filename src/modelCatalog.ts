import type {
  LoadedModelSummary as GptNeoLoadedModelSummary,
  LoaderProgress as GptNeoLoaderProgress,
  ModelConfig as GptNeoModelConfig,
  TensorVisualization as GptNeoTensorVisualization,
} from "./engine/src/gpt-neo/loader";
import type {
  Qwen2Config,
  Qwen2LoaderProgress,
} from "./engine/src/qwen2/loader";
import type { InferenceBackend, InferenceBackendPreference } from "./engine/src/runtime/webgpu";
import { currentModelExportName } from "./modelName";
import type { Qwen2TokenizerParts } from "./tokenizer";

export type ModelId = "tinystories" | "qwen";
export type PromptMode = "raw" | "qwen-chat";

export interface ModelCatalogEntry {
  id: ModelId;
  label: string;
  shortLabel: string;
  baseUrl: string;
  promptMode: PromptMode;
  backendPolicy: ModelBackendPolicy;
  tokenizerUrl?: string;
  ggufPath?: string;
  ggufFallbackUrls?: readonly string[];
}

export interface ModelBackendPolicy {
  defaultPreference: InferenceBackendPreference;
  webgpu: "preferred" | "required" | "unsupported";
  cpuFallback: boolean;
  minimumStorageBufferBindingSize?: number;
}

export interface AppTensorSummary {
  name: string;
  description: string;
  shape: readonly number[];
  byteOffset: number;
  byteLength: number;
  elementCount: number;
}

export type AppLoaderStage = GptNeoLoaderProgress["stage"] | Qwen2LoaderProgress["stage"];

export interface AppLoaderProgress {
  modelId: ModelId;
  stage: AppLoaderStage;
  message: string;
  source?: "network" | "cache";
  loadedBytes?: number;
  totalBytes?: number;
}

export interface AppLoadStep {
  key: string;
  stages: readonly AppLoaderStage[];
  label: string;
}

interface BaseAppLoadedModelSummary {
  modelId: ModelId;
  modelLabel: string;
  backend: InferenceBackend;
  architecture: string;
  dtype: string;
  tensorCount: number;
  totalByteLength: number;
  layers: number;
  hiddenSize: number;
  vocabularySize: number;
  maximumSequenceLength: number;
  tensors: AppTensorSummary[];
}

export interface TinyStoriesLoadedModelSummary extends BaseAppLoadedModelSummary {
  kind: "gpt-neo";
  modelId: "tinystories";
  scratchSequenceLength: number;
  config: GptNeoModelConfig;
  tensors: Array<AppTensorSummary & GptNeoTensorVisualization>;
}

export interface QwenLoadedModelSummary extends BaseAppLoadedModelSummary {
  kind: "qwen2";
  modelId: "qwen";
  keyValueHiddenSize: number;
  config: Qwen2Config;
}

export type AppLoadedModelSummary = TinyStoriesLoadedModelSummary | QwenLoadedModelSummary;

export type AppTokenizerPayload =
  | {
      kind: "qwen2-gguf";
      parts: Qwen2TokenizerParts;
    };

export type AppWorkerRequest =
  | {
      type: "load-model";
      requestId?: string;
      modelId: ModelId;
      baseUrl: string;
      configPath?: string;
      weightsIndexPath?: string;
      weightsBinaryPath?: string;
      scratchSequenceLength?: number;
      ggufPath?: string;
      ggufFallbackUrls?: readonly string[];
      backendPreference?: InferenceBackendPreference;
      webgpuRequired?: boolean;
    }
  | {
      type: "next-token";
      requestId?: string;
      modelId: ModelId;
      inputIds: number[];
      temperature?: number;
      topK?: number;
    };

export type AppWorkerResponse =
  | {
      type: "model-progress";
      requestId?: string;
      modelId: ModelId;
      progress: AppLoaderProgress;
    }
  | {
      type: "model-ready";
      requestId?: string;
      modelId: ModelId;
      backend: InferenceBackend;
      summary: AppLoadedModelSummary;
      tokenizer?: AppTokenizerPayload;
    }
  | {
      type: "next-token-result";
      requestId?: string;
      modelId: ModelId;
      tokenId: number;
    }
  | {
      type: "model-error";
      requestId?: string;
      modelId?: ModelId;
      error: string;
    };

const publicBaseUrl = import.meta.env.BASE_URL;

export const qwenModelFolderName = "qwen2.5-0.5b-instruct-q4_0";
export const qwenOfficialQ4_0GgufUrl =
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
export const defaultModelId: ModelId = "tinystories";

export const modelCatalog: Record<ModelId, ModelCatalogEntry> = {
  tinystories: {
    id: "tinystories",
    label: "TinyStories GPT-Neo",
    shortLabel: "TinyStories",
    baseUrl: `${publicBaseUrl}models/${currentModelExportName}/`,
    promptMode: "raw",
    backendPolicy: {
      defaultPreference: "auto",
      webgpu: "preferred",
      cpuFallback: true,
    },
    tokenizerUrl: `${publicBaseUrl}tokenizer/tinystories-tokenizer.json`,
  },
  qwen: {
    id: "qwen",
    label: "Qwen2.5 0.5B (Beta)",
    shortLabel: "Qwen",
    baseUrl: `${publicBaseUrl}models/${qwenModelFolderName}/`,
    promptMode: "qwen-chat",
    backendPolicy: {
      defaultPreference: "auto",
      webgpu: "required",
      cpuFallback: false,
      minimumStorageBufferBindingSize: 144_643_072,
    },
    ggufPath: "model.gguf",
    ggufFallbackUrls: [qwenOfficialQ4_0GgufUrl],
  },
};

export const modelOptions: readonly ModelCatalogEntry[] = [
  modelCatalog.tinystories,
  modelCatalog.qwen,
];

export const modelLoadSteps: Record<ModelId, readonly AppLoadStep[]> = {
  tinystories: [
    { key: "descriptors", stages: ["descriptors-download-started"], label: "Start descriptor downloads" },
    { key: "descriptors-downloaded", stages: ["descriptors-downloaded"], label: "Receive config and tensor index" },
    { key: "descriptors-validated", stages: ["descriptors-validated"], label: "Validate architecture and tensor metadata" },
    {
      key: "weights-download",
      stages: ["weights-download-started", "weights-download-progress", "weights-downloaded"],
      label: "Download raw FP32 weights",
    },
    { key: "weights-validated", stages: ["weights-validated"], label: "Validate weight buffer boundaries" },
    { key: "tensor-views", stages: ["tensor-views-created"], label: "Create zero-copy tensor views" },
    { key: "weights-bound", stages: ["weights-bound"], label: "Bind tensors into GPT-Neo layers" },
    { key: "scratch", stages: ["scratch-allocated"], label: "Allocate inference scratch buffers" },
    { key: "ready", stages: ["ready"], label: "Keep model resident in the worker" },
  ],
  qwen: [
    {
      key: "gguf-download",
      stages: ["gguf-download-started", "gguf-download-progress"],
      label: "Download GGUF model file",
    },
    { key: "gguf-downloaded", stages: ["gguf-downloaded"], label: "Receive GGUF bytes" },
    { key: "gguf-parsed", stages: ["gguf-parsed"], label: "Parse GGUF metadata and tensor table" },
    { key: "weights-bound", stages: ["weights-bound"], label: "Bind tensors into Qwen2 layers" },
    { key: "ready", stages: ["ready"], label: "Keep model resident in the worker" },
  ],
};

export function formatPromptForModel(modelId: ModelId, prompt: string): string {
  if (modelCatalog[modelId].promptMode !== "qwen-chat") {
    return prompt;
  }

  return (
    "<|im_start|>system\n" +
    "You are a helpful assistant.<|im_end|>\n" +
    `<|im_start|>user\n${prompt}<|im_end|>\n` +
    "<|im_start|>assistant\n"
  );
}

export function visibleGeneratedTextForModel(modelId: ModelId, decodedText: string): string {
  if (modelCatalog[modelId].promptMode !== "qwen-chat") {
    return decodedText;
  }

  return decodedText.replace(/<\|[^|]+?\|>/g, "");
}

export function normalizeGptNeoSummary(
  summary: GptNeoLoadedModelSummary,
  backend: InferenceBackend = "cpu",
): TinyStoriesLoadedModelSummary {
  return {
    ...summary,
    kind: "gpt-neo",
    modelId: "tinystories",
    modelLabel: modelCatalog.tinystories.label,
    backend,
    tensors: summary.tensors,
  };
}

export function stepIndexForProgressStage(
  stage: AppLoaderStage,
  steps: readonly AppLoadStep[],
): number {
  const index = steps.findIndex((step) => step.stages.includes(stage));
  return index >= 0 ? index : 0;
}

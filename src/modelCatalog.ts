import type {
  Qwen2Config,
  Qwen2LoaderProgress,
} from "./engine/src/qwen2/loader";
import type { InferenceBackend, InferenceBackendPreference } from "./engine/src/runtime/webgpu";
import type { Qwen2TokenizerParts } from "./tokenizer";

export type ModelId = "qwen" | "qwen_cpu_small";
export type PromptMode = "qwen-chat";

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

export type AppLoaderStage = Qwen2LoaderProgress["stage"];

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

export interface QwenLoadedModelSummary extends BaseAppLoadedModelSummary {
  kind: "qwen2";
  modelId: ModelId;
  keyValueHiddenSize: number;
  config: Qwen2Config;
}

export type AppLoadedModelSummary = QwenLoadedModelSummary;

export type AppInferencePhase = "prefill" | "decode";

export interface AppInferencePerformance {
  phase: AppInferencePhase;
  tokenCount: number;
  elapsedMs: number;
  tokensPerSecond: number;
}

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
      resetCache?: boolean;
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
      performance?: AppInferencePerformance;
    }
  | {
      type: "model-error";
      requestId?: string;
      modelId?: ModelId;
      error: string;
    };

const publicBaseUrl = import.meta.env.BASE_URL;

export const qwenModelFolderName = "qwen2.5-0.5b-instruct-q4_0";
export const qwenCpuSmallModelFolderName = "qwen2.5-0.5b-instruct-iq1_s";
export const qwenOfficialQ4_0GgufUrl =
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
export const qwenCpuSmallIQ1_SGgufUrl =
  "https://huggingface.co/legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF/resolve/main/Qwen2.5-0.5B-Instruct.IQ1_S.gguf";
export const defaultModelId: ModelId = "qwen";

export const modelCatalog: Record<ModelId, ModelCatalogEntry> = {
  qwen: {
    id: "qwen",
    label: "Qwen2.5 0.5B Q4_0",
    shortLabel: "Qwen Q4_0",
    baseUrl: `${publicBaseUrl}models/${qwenModelFolderName}/`,
    promptMode: "qwen-chat",
    backendPolicy: {
      defaultPreference: "auto",
      webgpu: "required",
      cpuFallback: false,
      minimumStorageBufferBindingSize: 144_643_072,
    },
    ggufPath: qwenOfficialQ4_0GgufUrl,
  },
  qwen_cpu_small: {
    id: "qwen_cpu_small",
    label: "Qwen2.5 0.5B IQ1_S CPU",
    shortLabel: "Qwen CPU",
    baseUrl: `${publicBaseUrl}models/${qwenCpuSmallModelFolderName}/`,
    promptMode: "qwen-chat",
    backendPolicy: {
      defaultPreference: "cpu",
      webgpu: "unsupported",
      cpuFallback: true,
    },
    ggufPath: qwenCpuSmallIQ1_SGgufUrl,
  },
};

export const modelOptions: readonly ModelCatalogEntry[] = [
  modelCatalog.qwen,
  modelCatalog.qwen_cpu_small,
];

export const modelLoadSteps: Record<ModelId, readonly AppLoadStep[]> = {
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
  qwen_cpu_small: [
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
  return (
    "<|im_start|>system\n" +
    "You are a helpful assistant.<|im_end|>\n" +
    `<|im_start|>user\n${prompt}<|im_end|>\n` +
    "<|im_start|>assistant\n"
  );
}

export function visibleGeneratedTextForModel(modelId: ModelId, decodedText: string): string {
  return decodedText.replace(/<\|[^|]+?\|>/g, "");
}

export function stepIndexForProgressStage(
  stage: AppLoaderStage,
  steps: readonly AppLoadStep[],
): number {
  const index = steps.findIndex((step) => step.stages.includes(stage));
  return index >= 0 ? index : 0;
}

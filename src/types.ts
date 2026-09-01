import type { InferenceBackend, ModelId } from "./models";

export type BroslmState = "idle" | "loading" | "ready" | "generating" | "error" | "disposed";
export type GenerationPhase = "prefill" | "decode";
export type GenerationFinishReason = "eos" | "max_tokens";
export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  readonly role: ChatRole;
  readonly content: string;
}

export type PromptInput = string | readonly ChatMessage[];

export interface TensorSummary {
  name: string;
  shape: readonly number[];
  byteLength: number;
}

export interface LoadedModelSummary {
  modelId: ModelId;
  modelLabel: string;
  backend: InferenceBackend;
  architecture: "qwen2";
  quantization: string;
  source: string;
  tensorCount: number;
  totalByteLength: number;
  layers: number;
  hiddenSize: number;
  vocabularySize: number;
  maximumSequenceLength: number;
  attentionHeads: number;
  keyValueHeads: number;
  keyValueHiddenSize: number;
  eosTokenId: number;
  tensors: readonly TensorSummary[];
}

export interface WebGpuLimitSummary {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}

export interface ModelSupport {
  modelId: ModelId;
  backend: InferenceBackend;
  supported: boolean;
  reason?: string;
  limits?: WebGpuLimitSummary;
}

export interface LoadModelOptions {
  signal?: AbortSignal;
}

export interface GenerationOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  signal?: AbortSignal;
}

export interface GenerationPerformance {
  phase: GenerationPhase;
  tokenCount: number;
  elapsedMs: number;
  tokensPerSecond: number;
}

export interface GenerationChunk {
  tokenId: number;
  tokenIndex: number;
  text: string;
  performance: GenerationPerformance;
}

export interface GenerationResult {
  text: string;
  tokenIds: readonly number[];
  inputTokenCount: number;
  finishReason: GenerationFinishReason;
  elapsedMs: number;
}

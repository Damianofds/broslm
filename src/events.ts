import type { InferenceBackend, ModelId } from "./models";
import type { GenerationFinishReason, GenerationPerformance, LoadedModelSummary } from "./types";

interface EventBase {
  timestampMs: number;
  operationId: string;
  modelId: ModelId;
}

export type BroslmEvent =
  | (EventBase & { type: "model-load-started" })
  | (EventBase & { type: "backend-selected"; backend: InferenceBackend })
  | (EventBase & { type: "model-download-started" })
  | (EventBase & {
      type: "model-download-progress";
      source: "cache" | "network";
      loadedBytes: number;
      totalBytes?: number;
    })
  | (EventBase & { type: "model-downloaded"; loadedBytes: number })
  | (EventBase & { type: "model-parsed" })
  | (EventBase & { type: "model-weights-bound" })
  | (EventBase & { type: "model-ready"; summary: LoadedModelSummary })
  | (EventBase & { type: "generation-started"; inputTokenCount: number; maxTokens: number })
  | (EventBase & {
      type: "generation-progress";
      generatedTokenCount: number;
      performance: GenerationPerformance;
    })
  | (EventBase & {
      type: "generation-completed";
      generatedTokenCount: number;
      finishReason: GenerationFinishReason;
      elapsedMs: number;
    })
  | (EventBase & { type: "operation-cancelled"; operation: "load" | "generation" })
  | (EventBase & {
      type: "operation-error";
      operation: "load" | "generation";
      code: string;
      message: string;
    });

export type BroslmEventListener = (event: BroslmEvent) => void;

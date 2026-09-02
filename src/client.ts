import { BroslmError } from "./errors";
import type { BroslmEnvironment } from "./environment";
import { createBroslmLogger, type BroslmLogger, type BroslmLogLevel } from "./logger";
import { modelCatalog, type ModelId } from "./models";
import { allocateQwen2ModelKvCache, resetQwen2ModelKvCache, type Qwen2ModelKvCache } from "./qwen2/attentionCache";
import type { GgufTensorInfo } from "./qwen2/gguf";
import {
  loadQwen2Model,
  type LoadedQwen2Model,
  type Qwen2LoaderProgress,
} from "./qwen2/loader";
import { qwen2NextTokenWithCacheGpu } from "./qwen2/model";
import { formatQwen2Prompt } from "./qwen2/chat";
import { createQwen2TokenizerFromGgufMetadata } from "./qwen2/tokenizer";
import {
  qwen2WebGpuCacheSequenceLength,
  qwen2WebGpuPrefillSafetyError,
  qwen2WebGpuSafetyLimits,
} from "./qwen2/webgpuSafety";
import {
  createWebGpuRuntime,
  destroyWebGpuRuntime,
  detectWebGpuSupport,
  type WebGpuRuntime,
} from "./runtime/webgpu";
import type { ByteLevelBpeTokenizer } from "./tokenizer";
import type {
  BroslmState,
  ChatMessage,
  GenerationChunk,
  GenerationFinishReason,
  GenerationOptions,
  GenerationPerformance,
  GenerationResult,
  LoadedModelSummary,
  LoadModelOptions,
  ModelSupport,
  PromptInput,
} from "./types";

export interface BroslmOptions {
  logLevel?: BroslmLogLevel;
}

export interface Broslm {
  readonly state: BroslmState;
  readonly loadedModel: LoadedModelSummary | null;
  checkModelSupport(modelId: ModelId): Promise<ModelSupport>;
  loadModel(modelId: ModelId, options?: LoadModelOptions): Promise<LoadedModelSummary>;
  countPromptTokens(prompt: string): number;
  countPromptTokens(messages: readonly ChatMessage[]): number;
  generate(prompt: string, options?: GenerationOptions): Promise<GenerationResult>;
  generate(messages: readonly ChatMessage[], options?: GenerationOptions): Promise<GenerationResult>;
  stream(prompt: string, options?: GenerationOptions): AsyncGenerator<GenerationChunk, void>;
  stream(messages: readonly ChatMessage[], options?: GenerationOptions): AsyncGenerator<GenerationChunk, void>;
  dispose(): void;
}

export interface BroslmClientDependencies {
  loadQwen2Model: typeof loadQwen2Model;
  createTokenizer: typeof createQwen2TokenizerFromGgufMetadata;
  allocateCache: typeof allocateQwen2ModelKvCache;
  resetCache: typeof resetQwen2ModelKvCache;
  nextToken: typeof qwen2NextTokenWithCacheGpu;
  detectWebGpuSupport: typeof detectWebGpuSupport;
  createWebGpuRuntime: typeof createWebGpuRuntime;
  destroyWebGpuRuntime: typeof destroyWebGpuRuntime;
}

const defaultDependencies: BroslmClientDependencies = {
  loadQwen2Model,
  createTokenizer: createQwen2TokenizerFromGgufMetadata,
  allocateCache: allocateQwen2ModelKvCache,
  resetCache: resetQwen2ModelKvCache,
  nextToken: qwen2NextTokenWithCacheGpu,
  detectWebGpuSupport,
  createWebGpuRuntime,
  destroyWebGpuRuntime,
};

export function createBroslmClient(
  environment: BroslmEnvironment,
  options: BroslmOptions = {},
  dependencies: BroslmClientDependencies = defaultDependencies,
): Broslm {
  return new DefaultBroslmClient(environment, options, dependencies);
}

class DefaultBroslmClient implements Broslm {
  private currentState: BroslmState = "idle";
  private summary: LoadedModelSummary | null = null;
  private model: LoadedQwen2Model | null = null;
  private modelId: ModelId | null = null;
  private tokenizer: ByteLevelBpeTokenizer | null = null;
  private cache: Qwen2ModelKvCache | null = null;
  private runtime: WebGpuRuntime | undefined;
  private readonly logger: BroslmLogger;
  private activeAbortController: AbortController | null = null;
  private lastGenerationResult: GenerationResult | null = null;

  constructor(
    private readonly environment: BroslmEnvironment,
    options: BroslmOptions,
    private readonly dependencies: BroslmClientDependencies,
  ) {
    this.logger = createBroslmLogger(options.logLevel);
  }

  get state(): BroslmState {
    return this.currentState;
  }

  get loadedModel(): LoadedModelSummary | null {
    return this.summary;
  }

  async checkModelSupport(modelId: ModelId): Promise<ModelSupport> {
    this.ensureNotDisposed();
    const descriptor = requireModelDescriptor(modelId);
    try {
      const target = await this.environment.getWebGpuTarget();
      const support = await this.dependencies.detectWebGpuSupport(target);
      if (!support.supported) {
        return {
          modelId,
          backend: "webgpu",
          supported: false,
          reason: support.reason,
          limits: support.limits,
        };
      }
      const requiredSize = descriptor.minimumStorageBufferBindingSize ?? 0;
      const availableSize = support.limits?.maxStorageBufferBindingSize ?? 0;
      if (requiredSize > availableSize) {
        return {
          modelId,
          backend: "webgpu",
          supported: false,
          reason:
            `This model needs maxStorageBufferBindingSize ${requiredSize}, ` +
            `but this adapter provides ${availableSize}.`,
          limits: support.limits,
        };
      }
      return { modelId, backend: "webgpu", supported: true, limits: support.limits };
    } catch (error: unknown) {
      return {
        modelId,
        backend: "webgpu",
        supported: false,
        reason: errorMessage(error),
      };
    }
  }

  async loadModel(
    modelId: ModelId,
    options: LoadModelOptions = {},
  ): Promise<LoadedModelSummary> {
    this.ensureNotDisposed();
    if (this.currentState === "loading" || this.currentState === "generating") {
      throw new BroslmError("INVALID_STATE", `Cannot load a model while broSLM is ${this.currentState}.`);
    }

    const descriptor = requireModelDescriptor(modelId);
    const operationId = createOperationId("load");
    const controller = createOperationController(options.signal);
    this.activeAbortController = controller;
    this.releaseLoadedModel();
    this.currentState = "loading";
    this.emit({ type: "model-load-started", modelId, operationId });

    try {
      throwIfAborted(controller.signal);
      const support = await this.checkModelSupport(modelId);
      if (!support.supported) {
        throw new BroslmError(
          "BACKEND_UNAVAILABLE",
          support.reason ?? "This model requires WebGPU, but no compatible adapter is available.",
        );
      }
      throwIfAborted(controller.signal);
      const target = await this.environment.getWebGpuTarget();
      const runtime = await this.dependencies.createWebGpuRuntime(target, {
        requiredStorageBufferBindingSize: descriptor.minimumStorageBufferBindingSize,
      });
      this.runtime = runtime;
      this.emit({
        type: "backend-selected",
        modelId,
        operationId,
        backend: descriptor.backend,
      });

      const model = await this.dependencies.loadQwen2Model({
        baseUrl: new URL(".", descriptor.source).toString(),
        ggufPath: descriptor.source,
        fetchImpl: this.environment.fetchImpl,
        signal: controller.signal,
        onProgress: (progress) => this.emitLoaderProgress(modelId, operationId, progress),
      });
      throwIfAborted(controller.signal);

      const tokenizer = this.dependencies.createTokenizer(model.gguf.metadata);
      const maximumCacheLength = qwen2WebGpuCacheSequenceLength(model.config.maximumSequenceLength);

      this.model = model;
      this.modelId = modelId;
      this.tokenizer = tokenizer;
      this.cache = this.dependencies.allocateCache(model.config, maximumCacheLength);
      this.summary = summarizeModel(modelId, model);
      this.currentState = "ready";
      this.emit({ type: "model-ready", modelId, operationId, summary: this.summary });
      return this.summary;
    } catch (error: unknown) {
      this.releaseLoadedModel();
      const aborted = controller.signal.aborted;
      this.currentState = "error";
      if (aborted) {
        this.emit({ type: "operation-cancelled", modelId, operationId, operation: "load" });
        throw new BroslmError("ABORTED", "Model loading was cancelled.", { cause: error });
      }
      const wrapped = asBroslmError(error, "MODEL_LOAD_FAILED");
      this.emit({
        type: "operation-error",
        modelId,
        operationId,
        operation: "load",
        code: wrapped.code,
        message: wrapped.message,
      });
      throw wrapped;
    } finally {
      if (this.activeAbortController === controller) {
        this.activeAbortController = null;
      }
    }
  }

  countPromptTokens(input: PromptInput): number {
    this.ensureReady();
    return this.tokenizer?.encode(formatQwen2Prompt(input)).length ?? 0;
  }

  async generate(input: PromptInput, options: GenerationOptions = {}): Promise<GenerationResult> {
    for await (const _chunk of this.streamInput(input, options)) {
      // The stream owns generation; the final result is retained when it completes.
    }
    if (!this.lastGenerationResult) {
      throw new BroslmError("GENERATION_FAILED", "Generation completed without a result.");
    }
    return this.lastGenerationResult;
  }

  stream(input: PromptInput, options: GenerationOptions = {}): AsyncGenerator<GenerationChunk, void> {
    return this.streamInput(input, options);
  }

  private async *streamInput(
    input: PromptInput,
    options: GenerationOptions,
  ): AsyncGenerator<GenerationChunk, void> {
    this.ensureReady();
    const formattedPrompt = formatQwen2Prompt(input);

    const model = this.model;
    const modelId = this.modelId;
    const tokenizer = this.tokenizer;
    const cache = this.cache;
    if (!model || !modelId || !tokenizer || !cache) {
      throw new BroslmError("INVALID_STATE", "Model and tokenizer must be ready before generation.");
    }

    const operationId = createOperationId("generation");
    const controller = createOperationController(options.signal);
    this.activeAbortController = controller;
    this.lastGenerationResult = null;
    this.currentState = "generating";
    const startedAt = nowMs();
    const inputIds = tokenizer.encode(formattedPrompt);
    const maxTokens = resolveMaximumNewTokens(options.maxTokens);
    const availableContext = model.config.maximumSequenceLength - inputIds.length;
    if (availableContext <= 0) {
      this.currentState = "ready";
      this.activeAbortController = null;
      throw new BroslmError("INVALID_ARGUMENT", "Prompt is longer than the model context window.");
    }

    if (this.runtime) {
      const safetyError = qwen2WebGpuPrefillSafetyError(inputIds.length);
      if (safetyError) {
        this.currentState = "ready";
        this.activeAbortController = null;
        throw new BroslmError("INVALID_ARGUMENT", safetyError);
      }
    }

    const targetTokens = Math.min(
      maxTokens,
      availableContext,
      Math.max(0, qwen2WebGpuSafetyLimits.maxSequenceTokens - inputIds.length),
    );
    if (targetTokens <= 0) {
      this.currentState = "ready";
      this.activeAbortController = null;
      throw new BroslmError("INVALID_ARGUMENT", "The model context has no room for generated tokens.");
    }

    this.dependencies.resetCache(cache);
    this.emit({
      type: "generation-started",
      modelId,
      operationId,
      inputTokenCount: inputIds.length,
      maxTokens: targetTokens,
    });

    const nextInputIds = [...inputIds];
    const generatedTokenIds: number[] = [];
    let generatedText = "";
    let finishReason: GenerationFinishReason = "max_tokens";
    let completed = false;

    try {
      for (let tokenIndex = 0; tokenIndex < targetTokens; tokenIndex += 1) {
        throwIfAborted(controller.signal);
        const phase = tokenIndex === 0 ? "prefill" : "decode";
        const profiledTokenCount = phase === "prefill" ? inputIds.length : 1;
        const tokenStartedAt = nowMs();
        const runtime = this.runtime;
        if (!runtime) {
          throw new BroslmError("BACKEND_UNAVAILABLE", "The WebGPU runtime is not initialized.");
        }
        const result = await this.dependencies.nextToken(
          model,
          nextInputIds,
          cache,
          runtime,
          this.logger,
          {
            temperature: options.temperature ?? 0.95,
            topK: options.topK ?? 10,
          },
        );
        throwIfAborted(controller.signal);
        const performance = createPerformance(phase, profiledTokenCount, nowMs() - tokenStartedAt);
        this.emit({
          type: "generation-progress",
          modelId,
          operationId,
          generatedTokenCount: generatedTokenIds.length,
          performance,
        });

        if (result.tokenId === model.config.eosTokenId || result.tokenId === tokenizer.eosTokenId) {
          finishReason = "eos";
          break;
        }

        nextInputIds.push(result.tokenId);
        generatedTokenIds.push(result.tokenId);
        generatedText = visibleGeneratedText(tokenizer.decode(generatedTokenIds));
        yield {
          tokenId: result.tokenId,
          tokenIndex,
          text: generatedText,
          performance,
        };
      }

      const elapsedMs = nowMs() - startedAt;
      this.lastGenerationResult = {
        text: generatedText,
        tokenIds: [...generatedTokenIds],
        inputTokenCount: inputIds.length,
        finishReason,
        elapsedMs,
      };
      completed = true;
      this.emit({
        type: "generation-completed",
        modelId,
        operationId,
        generatedTokenCount: generatedTokenIds.length,
        finishReason,
        elapsedMs,
      });
    } catch (error: unknown) {
      if (controller.signal.aborted) {
        this.emit({ type: "operation-cancelled", modelId, operationId, operation: "generation" });
        throw new BroslmError("ABORTED", "Generation was cancelled.", { cause: error });
      }
      const wrapped = asBroslmError(error, "GENERATION_FAILED");
      this.emit({
        type: "operation-error",
        modelId,
        operationId,
        operation: "generation",
        code: wrapped.code,
        message: wrapped.message,
      });
      throw wrapped;
    } finally {
      if (!completed && !controller.signal.aborted) {
        this.emit({ type: "operation-cancelled", modelId, operationId, operation: "generation" });
      }
      if ((this.currentState as BroslmState) !== "disposed") {
        this.currentState = "ready";
      }
      if (this.activeAbortController === controller) {
        this.activeAbortController = null;
      }
    }
  }

  dispose(): void {
    if (this.currentState === "disposed") {
      return;
    }
    this.activeAbortController?.abort(new BroslmError("DISPOSED", "broSLM was disposed."));
    this.activeAbortController = null;
    this.releaseLoadedModel();
    this.currentState = "disposed";
  }

  private ensureReady(): void {
    this.ensureNotDisposed();
    if (this.currentState !== "ready") {
      throw new BroslmError("INVALID_STATE", `broSLM must be ready, but it is ${this.currentState}.`);
    }
  }

  private ensureNotDisposed(): void {
    if (this.currentState === "disposed") {
      throw new BroslmError("DISPOSED", "broSLM has been disposed.");
    }
  }

  private releaseLoadedModel(): void {
    this.dependencies.destroyWebGpuRuntime(this.runtime);
    this.runtime = undefined;
    this.model = null;
    this.modelId = null;
    this.tokenizer = null;
    this.cache = null;
    this.summary = null;
    this.lastGenerationResult = null;
  }

  private emitLoaderProgress(
    modelId: ModelId,
    operationId: string,
    progress: Qwen2LoaderProgress,
  ): void {
    switch (progress.stage) {
      case "gguf-download-started":
        this.emit({ type: "model-download-started", modelId, operationId });
        break;
      case "gguf-download-progress":
        this.emit({
          type: "model-download-progress",
          modelId,
          operationId,
          source: progress.source ?? "network",
          loadedBytes: progress.loadedBytes ?? 0,
          totalBytes: progress.totalBytes,
        });
        break;
      case "gguf-downloaded":
        this.emit({
          type: "model-downloaded",
          modelId,
          operationId,
          loadedBytes: progress.loadedBytes ?? 0,
        });
        break;
      case "gguf-parsed":
        this.emit({ type: "model-parsed", modelId, operationId });
        break;
      case "weights-bound":
        this.emit({ type: "model-weights-bound", modelId, operationId });
        break;
      case "ready":
        break;
    }
  }

  private emit(event: { type: string; [key: string]: unknown }): void {
    const { type, ...context } = event;
    if (type.endsWith("-progress")) {
      this.logger.debug(type, context);
    } else if (type === "operation-error") {
      this.logger.error(type, context);
    } else {
      this.logger.info(type, context);
    }
  }
}

function summarizeModel(modelId: ModelId, model: LoadedQwen2Model): LoadedModelSummary {
  const descriptor = modelCatalog[modelId];
  return {
    modelId,
    modelLabel: descriptor.label,
    backend: descriptor.backend,
    architecture: model.config.architecture,
    quantization: describeGgufDtype(model),
    source: descriptor.source,
    tensorCount: model.gguf.tensors.size,
    totalByteLength: model.weightsBuffer.byteLength,
    layers: model.config.numberOfLayers,
    hiddenSize: model.config.hiddenSize,
    vocabularySize: model.config.vocabularySize,
    maximumSequenceLength: model.config.maximumSequenceLength,
    attentionHeads: model.config.numberOfHeads,
    keyValueHeads: model.config.numberOfKeyValueHeads,
    keyValueHiddenSize: model.config.keyValueHiddenSize,
    eosTokenId: model.config.eosTokenId,
    tensors: [...model.gguf.tensors.values()].map(summarizeTensor),
  };
}

function summarizeTensor(tensor: GgufTensorInfo) {
  return {
    name: tensor.name,
    shape: tensor.dimensions.length === 2
      ? [tensor.dimensions[1] ?? 0, tensor.dimensions[0] ?? 0]
      : [...tensor.dimensions],
    byteLength: tensor.byteLength,
  };
}

function describeGgufDtype(model: LoadedQwen2Model): string {
  const types = new Set([...model.gguf.tensors.values()].map((tensor) => tensor.type));
  return `GGUF ${[...types].map(ggmlTypeLabel).sort().join("/")}`;
}

function ggmlTypeLabel(type: number): string {
  return ({
    0: "F32",
    2: "Q4_0",
    8: "Q8_0",
  } as Record<number, string>)[type] ?? `type${type}`;
}

function requireModelDescriptor(modelId: ModelId) {
  const descriptor = modelCatalog[modelId];
  if (!descriptor) {
    throw new BroslmError("INVALID_ARGUMENT", `Unknown model id: ${String(modelId)}`);
  }
  return descriptor;
}

function visibleGeneratedText(decodedText: string): string {
  return decodedText.replace(/<\|[^|]+?\|>/g, "").replace(/\r\n?/g, "\n").replace(/\n+/g, "\n");
}

function resolveMaximumNewTokens(value: number | undefined): number {
  const resolved = value ?? 120;
  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new BroslmError("INVALID_ARGUMENT", `maxTokens must be a positive integer, got ${resolved}.`);
  }
  return resolved;
}

function createPerformance(
  phase: "prefill" | "decode",
  tokenCount: number,
  elapsedMs: number,
): GenerationPerformance {
  return {
    phase,
    tokenCount,
    elapsedMs,
    tokensPerSecond: elapsedMs > 0 ? tokenCount / (elapsedMs / 1000) : 0,
  };
}

function createOperationController(signal: AbortSignal | undefined): AbortController {
  const controller = new AbortController();
  if (signal?.aborted) {
    controller.abort(signal.reason);
  } else {
    signal?.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
  }
  return controller;
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw signal.reason ?? new BroslmError("ABORTED", "Operation was cancelled.");
  }
}

function asBroslmError(error: unknown, fallbackCode: "MODEL_LOAD_FAILED" | "GENERATION_FAILED") {
  return error instanceof BroslmError
    ? error
    : new BroslmError(fallbackCode, errorMessage(error), { cause: error });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function createOperationId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

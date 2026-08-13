import { useEffect, useMemo, useRef, useState } from "react";
import type {
  LoadedModelSummary,
  LoaderProgress,
  LoaderWorkerRequest,
  LoaderWorkerResponse,
} from "./engine/src/loader";
import { modelBaseUrl, tokenizerUrl } from "./modelExport";
import {
  loadByteLevelBpeTokenizer,
  type ByteLevelBpeTokenizer,
} from "./tokenizer";
import { createModelCacheFetch } from "./modelCache";

type LoadState = "idle" | "loading" | "ready" | "error";
type TokenizerState = "idle" | "loading" | "ready" | "error";
type GenerationState = "idle" | "generating" | "done" | "error";
type LoadFrame = "start" | "loading" | "config";

interface PendingNextTokenRequest {
  resolve: (tokenId: number) => void;
  reject: (error: Error) => void;
}

interface Step {
  stage: LoaderProgress["stage"];
  label: string;
}

const steps: readonly Step[] = [
  { stage: "descriptors-download-started", label: "Start descriptor downloads" },
  { stage: "descriptors-downloaded", label: "Receive config and tensor index" },
  { stage: "descriptors-validated", label: "Validate architecture and tensor metadata" },
  { stage: "weights-download-started", label: "Download raw FP32 weights" },
  { stage: "weights-validated", label: "Validate weight buffer boundaries" },
  { stage: "tensor-views-created", label: "Create zero-copy tensor views" },
  { stage: "weights-bound", label: "Bind tensors into GPT-Neo layers" },
  { stage: "scratch-allocated", label: "Allocate inference scratch buffers" },
  { stage: "ready", label: "Keep model resident in the worker" },
];

const defaultMaxNewTokens = 120;
const defaultTemperature = 0.95;
const defaultTopK = 10;

export default function App() {
  const workerRef = useRef<Worker | null>(null);
  const tokenizerRef = useRef<ByteLevelBpeTokenizer | null>(null);
  const cachedFetchRef = useRef<typeof fetch | null>(null);
  const pendingNextTokenRequestsRef = useRef<Map<string, PendingNextTokenRequest>>(new Map());
  const generationRunRef = useRef(0);
  const chatTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [tokenizerState, setTokenizerState] = useState<TokenizerState>("idle");
  const [progress, setProgress] = useState<LoaderProgress | null>(null);
  const [actualStepIndex, setActualStepIndex] = useState(-1);
  const [visibleStepIndex, setVisibleStepIndex] = useState(-1);
  const [summary, setSummary] = useState<LoadedModelSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tokenizerError, setTokenizerError] = useState<string | null>(null);
  const [chatText, setChatText] = useState("Once upon a time");
  const [maxNewTokens, setMaxNewTokens] = useState(defaultMaxNewTokens);
  const [temperature, setTemperature] = useState(defaultTemperature);
  const [topK, setTopK] = useState(defaultTopK);
  const [generationState, setGenerationState] = useState<GenerationState>("idle");
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [inputTokenCount, setInputTokenCount] = useState<number | null>(null);
  const [generatedTokenIds, setGeneratedTokenIds] = useState<number[]>([]);

  useEffect(() => {
    return () => {
      generationRunRef.current += 1;
      rejectPendingNextTokenRequests(
        pendingNextTokenRequestsRef.current,
        new Error("Component unmounted"),
      );
      workerRef.current?.terminate();
    };
  }, []);

  useEffect(() => {
    if (visibleStepIndex >= actualStepIndex) {
      return;
    }

    const delayMs = visibleStepIndex < 0 ? 0 : 500;
    const timeout = window.setTimeout(() => {
      setVisibleStepIndex((current) => Math.min(current + 1, actualStepIndex));
    }, delayMs);

    return () => window.clearTimeout(timeout);
  }, [actualStepIndex, visibleStepIndex]);

  useEffect(() => {
    const textarea = chatTextareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [chatText]);

  const chatTokenPreview = useMemo(() => {
    if (!tokenizerRef.current || chatText.length === 0) {
      return null;
    }

    try {
      return tokenizerRef.current.encode(chatText).length;
    } catch {
      return null;
    }
  }, [chatText, tokenizerState]);

  const canGenerate =
    loadState === "ready" &&
    tokenizerState === "ready" &&
    generationState !== "generating" &&
    chatText.length > 0;

  const loadFrame = resolveLoadFrame(loadState, summary, visibleStepIndex);
  const canEnterChat =
    loadFrame === "config" && loadState === "ready" && tokenizerState === "ready" && summary !== null;

  function loadModel() {
    if (loadState === "loading") {
      return;
    }

    generationRunRef.current += 1;
    rejectPendingNextTokenRequests(
      pendingNextTokenRequestsRef.current,
      new Error("Model is reloading"),
    );
    setLoadState("loading");
    setProgress(null);
    setActualStepIndex(-1);
    setVisibleStepIndex(-1);
    setSummary(null);
    setError(null);
    setGenerationState("idle");
    setGenerationError(null);
    setGeneratedTokenIds([]);
    setInputTokenCount(null);
    void ensureTokenizerLoaded();

    workerRef.current?.terminate();
    const worker = new Worker(new URL("./modelWorker.ts", import.meta.url), {
      type: "module",
      name: "broslm-inference-worker",
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<LoaderWorkerResponse>) => {
      const message = event.data;

      if (message.type === "model-progress") {
        setProgress(message.progress);
        setActualStepIndex((current) =>
          Math.max(current, stepIndexForStage(message.progress.stage)),
        );
        return;
      }

      if (message.type === "model-ready") {
        setSummary(message.summary);
        setActualStepIndex(steps.length - 1);
        setLoadState("ready");
        return;
      }

      if (message.type === "next-token-result") {
        const pending = message.requestId
          ? pendingNextTokenRequestsRef.current.get(message.requestId)
          : undefined;
        if (pending && message.requestId) {
          pendingNextTokenRequestsRef.current.delete(message.requestId);
          pending.resolve(message.tokenId);
        }
        return;
      }

      if (message.requestId) {
        const pending = pendingNextTokenRequestsRef.current.get(message.requestId);
        if (pending) {
          pendingNextTokenRequestsRef.current.delete(message.requestId);
          pending.reject(new Error(message.error));
          return;
        }
      }

      setError(message.error);
      setLoadState("error");
    };

    worker.onerror = (event) => {
      const message = event.message || "The inference worker failed while loading the model.";
      rejectPendingNextTokenRequests(pendingNextTokenRequestsRef.current, new Error(message));
      setError(message);
      setLoadState("error");
    };

    worker.postMessage({
      type: "load-model",
      requestId: createRequestId(),
      baseUrl: new URL(modelBaseUrl, window.location.href).toString(),
      scratchSequenceLength: 256,
    } satisfies LoaderWorkerRequest);
  }

  async function ensureTokenizerLoaded() {
    if (tokenizerRef.current || tokenizerState === "loading") {
      return;
    }

    setTokenizerState("loading");
    setTokenizerError(null);
    try {
      tokenizerRef.current = await loadByteLevelBpeTokenizer(tokenizerUrl, getCachedFetch());
      setTokenizerState("ready");
    } catch (loadError: unknown) {
      tokenizerRef.current = null;
      setTokenizerState("error");
      setTokenizerError(loadError instanceof Error ? loadError.message : String(loadError));
    }
  }

  function getCachedFetch(): typeof fetch {
    cachedFetchRef.current ??= createModelCacheFetch();
    return cachedFetchRef.current;
  }

  async function generateCompletion() {
    const tokenizer = tokenizerRef.current;
    if (!tokenizer || loadState !== "ready" || !summary) {
      setGenerationState("error");
      setGenerationError("Model and tokenizer must be ready before generation.");
      return;
    }

    if (chatText.length === 0) {
      setGenerationState("error");
      setGenerationError("Prompt is empty.");
      return;
    }

    const baseText = normalizeLineBreaks(chatText);
    setChatText(baseText);
    const inputIds = tokenizer.encode(baseText);
    const availableNewTokens = summary.config.maximumSequenceLength - inputIds.length;
    if (availableNewTokens <= 0) {
      setGenerationState("error");
      setGenerationError("Prompt is longer than the model context window.");
      return;
    }

    const runId = generationRunRef.current + 1;
    generationRunRef.current = runId;
    const nextInputIds = [...inputIds];
    const nextGeneratedTokenIds: number[] = [];
    const targetNewTokens = Math.min(maxNewTokens, availableNewTokens);

    setInputTokenCount(inputIds.length);
    setGeneratedTokenIds([]);
    setGenerationError(null);
    setGenerationState("generating");

    try {
      for (let tokenIndex = 0; tokenIndex < targetNewTokens; tokenIndex += 1) {
        const tokenId = await requestNextToken(nextInputIds, {
          temperature,
          topK,
        });
        if (generationRunRef.current !== runId) {
          return;
        }
        if (tokenId === summary.config.eosTokenId || tokenId === tokenizer.eosTokenId) {
          break;
        }

        nextInputIds.push(tokenId);
        nextGeneratedTokenIds.push(tokenId);
        setGeneratedTokenIds([...nextGeneratedTokenIds]);
        setChatText(normalizeLineBreaks(baseText + tokenizer.decode(nextGeneratedTokenIds)));
      }

      if (generationRunRef.current === runId) {
        setGenerationState("done");
      }
    } catch (generationError: unknown) {
      if (generationRunRef.current !== runId) {
        return;
      }
      setGenerationState("error");
      setGenerationError(
        generationError instanceof Error ? generationError.message : String(generationError),
      );
    }
  }

  function stopGeneration() {
    generationRunRef.current += 1;
    rejectPendingNextTokenRequests(
      pendingNextTokenRequestsRef.current,
      new Error("Generation stopped"),
    );
    setGenerationState("done");
  }

  function requestNextToken(
    inputIds: number[],
    options: { temperature: number; topK: number },
  ): Promise<number> {
    const worker = workerRef.current;
    if (!worker) {
      return Promise.reject(new Error("Inference worker is not running."));
    }

    const requestId = createRequestId();
    const request: LoaderWorkerRequest = {
      type: "next-token",
      requestId,
      inputIds,
      temperature: options.temperature,
      topK: options.topK,
    };

    return new Promise((resolve, reject) => {
      pendingNextTokenRequestsRef.current.set(requestId, { resolve, reject });
      worker.postMessage(request);
    });
  }

  return (
    <main className="page-shell">
      <OverviewSection />

      <LoadModelSection
        error={error}
        frame={loadFrame}
        loadState={loadState}
        progress={progress}
        summary={summary}
        tokenizerError={tokenizerError}
        visibleStepIndex={visibleStepIndex}
        canEnterChat={canEnterChat}
        onLoadModel={loadModel}
      />

      {canEnterChat && (
        <ChatSection
          canGenerate={canGenerate}
          chatText={chatText}
          generatedTokenCount={generatedTokenIds.length}
          generationError={generationError}
          generationState={generationState}
          loadState={loadState}
          maxNewTokens={maxNewTokens}
          temperature={temperature}
          textareaRef={chatTextareaRef}
          tokenCount={chatTokenPreview ?? inputTokenCount}
          tokenizerState={tokenizerState}
          topK={topK}
          onChatTextChange={(nextText) => setChatText(normalizeLineBreaks(nextText))}
          onGenerate={() => {
            void generateCompletion();
          }}
          onMaxNewTokensChange={setMaxNewTokens}
          onTemperatureChange={setTemperature}
          onStop={stopGeneration}
          onTopKChange={setTopK}
        />
      )}
    </main>
  );
}

function OverviewSection() {
  return (
    <section className="landing-section overview-section" id="overview">
      <div className="section-inner overview-inner">
        <h1>
          broSLM
          <span>browser small language model</span>
        </h1>
        <p className="overview-copy">
          A raw neural model running locally in the browser. Load the weights, chat without a
          server-side inference API.
        </p>
      </div>
      <ScrollCue href="#load-model" label="Discover more" />
    </section>
  );
}

function LoadModelSection({
  error,
  frame,
  loadState,
  progress,
  summary,
  tokenizerError,
  visibleStepIndex,
  canEnterChat,
  onLoadModel,
}: {
  error: string | null;
  frame: LoadFrame;
  loadState: LoadState;
  progress: LoaderProgress | null;
  summary: LoadedModelSummary | null;
  tokenizerError: string | null;
  visibleStepIndex: number;
  canEnterChat: boolean;
  onLoadModel: () => void;
}) {
  return (
    <section className="landing-section load-section" id="load-model">
      <div className="section-inner load-inner">
        <h2>{loadFrameTitle(frame, loadState)}</h2>
        <div className={`load-frame ${frame}-frame`}>
          {frame === "start" && (
            <StartLoadFrame
              error={error}
              loadState={loadState}
              tokenizerError={tokenizerError}
              onLoadModel={onLoadModel}
            />
          )}
          {frame === "loading" && (
            <LoadingFrame
              loadState={loadState}
              progress={progress}
              visibleStepIndex={visibleStepIndex}
            />
          )}
          {frame === "config" && summary && <ConfigFrame summary={summary} onLoadModel={onLoadModel} />}
        </div>
      </div>
      {canEnterChat && <ScrollCue href="#chat" label="Try the chat demo" />}
    </section>
  );
}

function ScrollCue({ href, label }: { href: string; label: string }) {
  return (
    <a className="scroll-cue" href={href}>
      <span className="scroll-cue-label">{label}</span>
      <span className="scroll-cue-arrow" aria-hidden="true" />
    </a>
  );
}

function StartLoadFrame({
  error,
  loadState,
  tokenizerError,
  onLoadModel,
}: {
  error: string | null;
  loadState: LoadState;
  tokenizerError: string | null;
  onLoadModel: () => void;
}) {
  return (
    <div className="start-load-frame">
      <button
        className="load-button"
        disabled={loadState === "loading"}
        onClick={onLoadModel}
        type="button"
      >
        Start
      </button>
      <p className="frame-copy">
        The browser will fetch the model weights, validate them, and keep the network quiet after
        the worker is ready.
      </p>
      {error && <p className="error-message">{error}</p>}
      {tokenizerError && <p className="error-message">{tokenizerError}</p>}
    </div>
  );
}

function LoadingFrame({
  loadState,
  progress,
  visibleStepIndex,
}: {
  loadState: LoadState;
  progress: LoaderProgress | null;
  visibleStepIndex: number;
}) {
  return (
    <div className="loading-frame-inner">
      <div className="progress-track" aria-label="Model loading progress">
        <div className="progress-fill" style={{ width: `${stepPercent(visibleStepIndex)}%` }} />
      </div>

      <p className="current-message">
        {currentStepMessage(visibleStepIndex, loadState)}
        {progress?.loadedBytes && progress.totalBytes
          ? ` / ${formatBytes(progress.loadedBytes)} of ${formatBytes(progress.totalBytes)}`
          : ""}
      </p>

      <ol className="step-list">
        {steps.map((step, index) => (
          <li className={stepClassName(index, visibleStepIndex)} key={step.stage}>
            <span className="step-index">{String(index + 1).padStart(2, "0")}</span>
            <span>{step.label}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ConfigFrame({
  summary,
  onLoadModel,
}: {
  summary: LoadedModelSummary;
  onLoadModel: () => void;
}) {
  return (
    <div className="config-frame-inner">
      <section className="config-panel">
        <h3>Model config</h3>
        <div className="config-grid">
          {modelConfigRows(summary).map((row) => (
            <ConfigItem key={row.label} label={row.label} value={row.value} />
          ))}
        </div>
        <div className="attention-strip" aria-label="Attention layer types">
          {summary.config.attentionLayers.map((kind, index) => (
            <span className={kind} key={`${kind}-${index}`} title={`Layer ${index}: ${kind}`}>
              {index}
            </span>
          ))}
        </div>
      </section>

      <section className="config-panel">
        <h3>Tensor config</h3>
        <div className="config-grid tensor-config-grid">
          {tensorConfigRows(summary).map((row) => (
            <ConfigItem key={row.label} label={row.label} value={row.value} />
          ))}
        </div>
      </section>

      <button className="reload-link" onClick={onLoadModel} type="button">
        Reload model
      </button>
    </div>
  );
}

function ChatSection({
  canGenerate,
  chatText,
  generatedTokenCount,
  generationError,
  generationState,
  loadState,
  maxNewTokens,
  temperature,
  textareaRef,
  tokenCount,
  tokenizerState,
  topK,
  onChatTextChange,
  onGenerate,
  onMaxNewTokensChange,
  onTemperatureChange,
  onStop,
  onTopKChange,
}: {
  canGenerate: boolean;
  chatText: string;
  generatedTokenCount: number;
  generationError: string | null;
  generationState: GenerationState;
  loadState: LoadState;
  maxNewTokens: number;
  temperature: number;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  tokenCount: number | null;
  tokenizerState: TokenizerState;
  topK: number;
  onChatTextChange: (value: string) => void;
  onGenerate: () => void;
  onMaxNewTokensChange: (value: number) => void;
  onTemperatureChange: (value: number) => void;
  onStop: () => void;
  onTopKChange: (value: number) => void;
}) {
  return (
    <section className="landing-section chat-section" id="chat">
      <div className="section-inner chat-inner">
        <div className="chat-heading">
          <h2>Try the chat demo</h2>
          <div className="status-pills">
            <span>{statusTitle(loadState)}</span>
            <span>{tokenizerStatusTitle(tokenizerState)}</span>
          </div>
        </div>

        <form
          className="chat-form"
          onSubmit={(event) => {
            event.preventDefault();
            onGenerate();
          }}
        >
          <label className="chat-textarea-label">
            <span>Prompt and completion</span>
            <textarea
              ref={textareaRef}
              value={chatText}
              onChange={(event) => onChatTextChange(event.target.value)}
              spellCheck={false}
            />
          </label>

          <div className="chat-controls">
            <GenerationControl
              label="New tokens"
              max={120}
              min={1}
              onChange={(nextValue) => onMaxNewTokensChange(clampTokenLimit(nextValue))}
              step={1}
              value={maxNewTokens}
              valueLabel={String(maxNewTokens)}
            />
            <GenerationControl
              label="Temperature"
              max={2}
              min={0}
              onChange={(nextValue) => onTemperatureChange(clampTemperature(nextValue))}
              step={0.05}
              value={temperature}
              valueLabel={temperature.toFixed(2)}
            />
            <GenerationControl
              label="Top-k"
              max={200}
              min={1}
              onChange={(nextValue) => onTopKChange(clampTopK(nextValue))}
              step={1}
              value={topK}
              valueLabel={String(topK)}
            />
            <div className="token-stats">
              <span>{tokenCount ?? "-"} tokens</span>
              <span>{generatedTokenCount} generated</span>
              <span>{generationTitle(generationState)}</span>
            </div>
            <div className="chat-actions">
              <button className="generate-button" disabled={!canGenerate} type="submit">
                Generate
              </button>
              <button
                className="stop-button"
                disabled={generationState !== "generating"}
                onClick={onStop}
                type="button"
              >
                Stop
              </button>
            </div>
          </div>

          {generationError && <p className="error-message">{generationError}</p>}
        </form>
      </div>
    </section>
  );
}

function GenerationControl({
  label,
  max,
  min,
  onChange,
  step,
  value,
  valueLabel,
}: {
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step: number;
  value: number;
  valueLabel: string;
}) {
  return (
    <label className="token-limit-field">
      <span className="token-limit-header">
        <span className="token-limit-label">{label}</span>
        <strong>{valueLabel}</strong>
      </span>
      <input
        aria-label={label}
        aria-valuetext={valueLabel}
        max={max}
        min={min}
        onChange={(event) => onChange(event.currentTarget.valueAsNumber)}
        step={step}
        type="range"
        value={value}
      />
    </label>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="config-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function resolveLoadFrame(
  loadState: LoadState,
  summary: LoadedModelSummary | null,
  visibleStepIndex: number,
): LoadFrame {
  if (summary && visibleStepIndex >= steps.length - 1) {
    return "config";
  }
  if (loadState === "loading" || summary) {
    return "loading";
  }
  return "start";
}

function loadFrameTitle(frame: LoadFrame, loadState: LoadState): string {
  if (frame === "config") {
    return "Model loaded";
  }
  if (frame === "loading") {
    return "Loading model";
  }
  if (loadState === "error") {
    return "Load failed";
  }
  return "Load model";
}

function statusTitle(loadState: LoadState): string {
  switch (loadState) {
    case "loading":
      return "Loading";
    case "ready":
      return "Model ready";
    case "error":
      return "Load failed";
    default:
      return "Model idle";
  }
}

function tokenizerStatusTitle(tokenizerState: TokenizerState): string {
  switch (tokenizerState) {
    case "loading":
      return "Tokenizer loading";
    case "ready":
      return "Tokenizer ready";
    case "error":
      return "Tokenizer failed";
    default:
      return "Tokenizer idle";
  }
}

function generationTitle(generationState: GenerationState): string {
  switch (generationState) {
    case "generating":
      return "Generating";
    case "done":
      return "Done";
    case "error":
      return "Error";
    default:
      return "Ready";
  }
}

function clampTokenLimit(value: number): number {
  if (!Number.isFinite(value)) {
    return defaultMaxNewTokens;
  }
  return Math.max(1, Math.min(120, Math.round(value)));
}

function clampTemperature(value: number): number {
  if (!Number.isFinite(value)) {
    return defaultTemperature;
  }
  return Math.max(0, Math.min(2, Math.round(value * 100) / 100));
}

function clampTopK(value: number): number {
  if (!Number.isFinite(value)) {
    return defaultTopK;
  }
  return Math.max(1, Math.min(200, Math.round(value)));
}

function normalizeLineBreaks(value: string): string {
  return value.replace(/\r\n?/g, "\n").replace(/\n+/g, "\n");
}

function createRequestId(): string {
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi?.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }

  if (typeof cryptoApi?.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    cryptoApi.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  return `request-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function rejectPendingNextTokenRequests(
  requests: Map<string, PendingNextTokenRequest>,
  error: Error,
): void {
  for (const request of requests.values()) {
    request.reject(error);
  }
  requests.clear();
}

function stepIndexForStage(stage: LoaderProgress["stage"]): number {
  switch (stage) {
    case "descriptors-download-started":
      return 0;
    case "descriptors-downloaded":
      return 1;
    case "descriptors-validated":
      return 2;
    case "weights-download-started":
    case "weights-download-progress":
    case "weights-downloaded":
      return 3;
    case "weights-validated":
      return 4;
    case "tensor-views-created":
      return 5;
    case "weights-bound":
      return 6;
    case "scratch-allocated":
      return 7;
    case "ready":
      return 8;
    default:
      return 0;
  }
}

function stepClassName(index: number, visibleStepIndex: number): string {
  if (index < visibleStepIndex) {
    return "done";
  }
  if (index === visibleStepIndex) {
    return "active";
  }
  return "";
}

function stepPercent(visibleStepIndex: number): number {
  if (visibleStepIndex < 0) {
    return 0;
  }
  return Math.min(100, ((visibleStepIndex + 1) / steps.length) * 100);
}

function currentStepMessage(visibleStepIndex: number, loadState: LoadState): string {
  if (visibleStepIndex >= 0) {
    return steps[visibleStepIndex]?.label ?? statusTitle(loadState);
  }
  return statusTitle(loadState);
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }

  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function modelConfigRows(summary: LoadedModelSummary): Array<{ label: string; value: string }> {
  const { config } = summary;
  return [
    { label: "Architecture", value: summary.architecture },
    { label: "Dtype", value: summary.dtype },
    { label: "Layers", value: formatInteger(summary.layers) },
    { label: "Hidden size", value: formatInteger(summary.hiddenSize) },
    { label: "Vocabulary", value: formatInteger(summary.vocabularySize) },
    { label: "Max sequence", value: formatInteger(config.maximumSequenceLength) },
    { label: "Intermediate", value: formatInteger(config.intermediateSize) },
    { label: "Heads", value: formatInteger(config.numberOfHeads) },
    { label: "Head dim", value: formatInteger(config.headDimension) },
    { label: "Activation", value: config.activation },
    { label: "Window size", value: formatInteger(config.windowSize) },
    { label: "Layer norm eps", value: String(config.layerNormEpsilon) },
    { label: "BOS / EOS / PAD", value: `${config.bosTokenId} / ${config.eosTokenId} / ${config.padTokenId ?? "null"}` },
    { label: "Tied embeddings", value: config.tiedWordEmbeddings ? "yes" : "no" },
  ];
}

function tensorConfigRows(summary: LoadedModelSummary): Array<{ label: string; value: string }> {
  const tokenEmbedding = tensorByName(summary, "transformer.wte.weight");
  const positionEmbedding = tensorByName(summary, "transformer.wpe.weight");
  const lmHead = tensorByName(summary, "lm_head.weight");
  const finalNorm = tensorByName(summary, "transformer.ln_f.weight");

  return [
    { label: "Tensor count", value: formatInteger(summary.tensorCount) },
    { label: "Weight bytes", value: formatBytes(summary.totalByteLength) },
    { label: "Scratch sequence", value: formatInteger(summary.scratchSequenceLength) },
    { label: "Token embedding", value: tokenEmbedding ? formatShape(tokenEmbedding.shape) : "-" },
    { label: "Position embedding", value: positionEmbedding ? formatShape(positionEmbedding.shape) : "-" },
    { label: "LM head", value: lmHead ? formatShape(lmHead.shape) : "-" },
    { label: "Final norm", value: finalNorm ? formatShape(finalNorm.shape) : "-" },
    { label: "Largest tensor", value: largestTensorLabel(summary) },
  ];
}

function tensorByName(summary: LoadedModelSummary, name: string) {
  return summary.tensors.find((tensor) => tensor.name === name) ?? null;
}

function largestTensorLabel(summary: LoadedModelSummary): string {
  const tensor = [...summary.tensors].sort((left, right) => right.byteLength - left.byteLength)[0];
  if (!tensor) {
    return "-";
  }
  return `${tensor.name} ${formatShape(tensor.shape)}`;
}

function formatShape(shape: readonly number[]): string {
  return `[${shape.map(formatInteger).join(", ")}]`;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

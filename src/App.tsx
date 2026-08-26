import { useEffect, useMemo, useRef, useState } from "react";
import { detectWebGpuSupport } from "./engine/src/runtime/webgpu";
import {
  qwen2WebGpuPrefillSafetyError,
  qwen2WebGpuSafetyLimits,
} from "./engine/src/qwen2/webgpuSafety";
import {
  createQwen2ByteLevelBpeTokenizer,
  loadByteLevelBpeTokenizer,
  type ByteLevelBpeTokenizer,
} from "./tokenizer";
import { createModelCacheFetch } from "./modelCache";
import {
  defaultModelId,
  formatPromptForModel,
  modelCatalog,
  modelLoadSteps,
  modelOptions,
  stepIndexForProgressStage,
  visibleGeneratedTextForModel,
  type AppInferencePerformance,
  type AppLoadedModelSummary,
  type AppLoaderProgress,
  type AppLoadStep,
  type AppWorkerRequest,
  type AppWorkerResponse,
  type ModelCatalogEntry,
  type ModelId,
} from "./modelCatalog";

type LoadState = "idle" | "loading" | "ready" | "error";
type TokenizerState = "idle" | "loading" | "ready" | "error";
type GenerationState = "idle" | "generating" | "done" | "error";
type LoadFrame = "start" | "loading" | "transition" | "config";
type WebGpuAvailability = "checking" | "available" | "unavailable";

interface PendingNextTokenRequest {
  resolve: (tokenId: number) => void;
  reject: (error: Error) => void;
}

interface GenerationThroughput {
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
}

interface DecodeThroughputAccumulator {
  tokenCount: number;
  elapsedMs: number;
}

const defaultMaxNewTokens = 120;
const defaultTemperature = 0.95;
const defaultTopK = 10;
const configTransitionMs = 800;
const transitionTiles = Array.from({ length: 18 }, (_, index) => index);

export default function App() {
  const workerRef = useRef<Worker | null>(null);
  const tokenizerRef = useRef<ByteLevelBpeTokenizer | null>(null);
  const tokenizerModelIdRef = useRef<ModelId | null>(null);
  const selectedModelIdRef = useRef<ModelId>(defaultModelId);
  const cachedFetchRef = useRef<typeof fetch | null>(null);
  const pendingNextTokenRequestsRef = useRef<Map<string, PendingNextTokenRequest>>(new Map());
  const generationRunRef = useRef(0);
  const qwenGenerationInFlightRef = useRef(false);
  const decodeThroughputRef = useRef<DecodeThroughputAccumulator>({ tokenCount: 0, elapsedMs: 0 });
  const chatTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  const [selectedModelId, setSelectedModelId] = useState<ModelId>(defaultModelId);
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [tokenizerState, setTokenizerState] = useState<TokenizerState>("idle");
  const [progress, setProgress] = useState<AppLoaderProgress | null>(null);
  const [transientProgressMessage, setTransientProgressMessage] = useState<string | null>(null);
  const [configRevealReady, setConfigRevealReady] = useState(false);
  const [actualStepIndex, setActualStepIndex] = useState(-1);
  const [visibleStepIndex, setVisibleStepIndex] = useState(-1);
  const [summary, setSummary] = useState<AppLoadedModelSummary | null>(null);
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
  const [generationThroughput, setGenerationThroughput] = useState<GenerationThroughput>({
    prefillTokensPerSecond: null,
    decodeTokensPerSecond: null,
  });
  const [webgpuAvailability, setWebgpuAvailability] = useState<WebGpuAvailability>("checking");
  const [webgpuMaxStorageBufferBindingSize, setWebgpuMaxStorageBufferBindingSize] =
    useState<number | null>(null);
  const activeSteps = modelLoadSteps[selectedModelId];

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
    let cancelled = false;
    void detectWebGpuSupport().then((support) => {
      if (!cancelled) {
        setWebgpuAvailability(support.supported ? "available" : "unavailable");
        setWebgpuMaxStorageBufferBindingSize(
          support.limits?.maxStorageBufferBindingSize ?? null,
        );
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    selectedModelIdRef.current = selectedModelId;
  }, [selectedModelId]);

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
    if (!summary || visibleStepIndex < activeSteps.length - 1 || configRevealReady) {
      return;
    }

    const timeout = window.setTimeout(() => {
      setConfigRevealReady(true);
    }, configTransitionMs);

    return () => window.clearTimeout(timeout);
  }, [activeSteps.length, configRevealReady, summary, visibleStepIndex]);

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
      return tokenizerRef.current.encode(
        formatPromptForModel(selectedModelId, normalizeLineBreaks(chatText)),
      ).length;
    } catch {
      return null;
    }
  }, [chatText, selectedModelId, tokenizerState]);

  const canGenerate =
    loadState === "ready" &&
    tokenizerState === "ready" &&
    generationState !== "generating" &&
    chatText.length > 0;

  const loadFrame = resolveLoadFrame(
    loadState,
    summary,
    visibleStepIndex,
    configRevealReady,
    activeSteps,
  );
  const canEnterChat =
    loadFrame === "config" && loadState === "ready" && tokenizerState === "ready" && summary !== null;

  function loadModel() {
    if (loadState === "loading") {
      return;
    }

    const model = modelCatalog[selectedModelId];
    if (!modelCanRun(model, webgpuAvailability, webgpuMaxStorageBufferBindingSize)) {
      setError(
        modelUnavailableMessage(model, webgpuAvailability, webgpuMaxStorageBufferBindingSize),
      );
      setLoadState("error");
      return;
    }
    generationRunRef.current += 1;
    rejectPendingNextTokenRequests(
      pendingNextTokenRequestsRef.current,
      new Error("Model is reloading"),
    );
    tokenizerRef.current = null;
    tokenizerModelIdRef.current = null;
    setLoadState("loading");
    setTokenizerState("loading");
    setProgress(null);
    setTransientProgressMessage(null);
    setConfigRevealReady(false);
    setActualStepIndex(-1);
    setVisibleStepIndex(-1);
    setSummary(null);
    setError(null);
    setGenerationState("idle");
    setGenerationError(null);
    setGeneratedTokenIds([]);
    setInputTokenCount(null);
    resetGenerationThroughput();
    if (selectedModelId === "tinystories") {
      void ensureTokenizerLoaded(selectedModelId);
    }

    workerRef.current?.terminate();
    const worker = new Worker(new URL("./modelWorker.ts", import.meta.url), {
      type: "module",
      name: "broslm-inference-worker",
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<AppWorkerResponse>) => {
      const message = event.data;
      if (message.modelId && message.modelId !== selectedModelIdRef.current) {
        return;
      }

      if (message.type === "model-progress") {
        setProgress(message.progress);
        if (message.progress.source === "cache") {
          setTransientProgressMessage(message.progress.message);
          window.setTimeout(() => {
            setTransientProgressMessage((current) =>
              current === message.progress.message ? null : current,
            );
          }, 1200);
        }
        setActualStepIndex((current) =>
          Math.max(
            current,
            stepIndexForProgressStage(
              message.progress.stage,
              modelLoadSteps[message.modelId],
            ),
          ),
        );
        return;
      }

      if (message.type === "model-ready") {
        setSummary(message.summary);
        setActualStepIndex(modelLoadSteps[message.modelId].length - 1);
        setLoadState("ready");
        if (message.modelId === "qwen") {
          try {
            if (message.tokenizer?.kind !== "qwen2-gguf") {
              throw new Error("Qwen tokenizer metadata was missing from the worker response.");
            }
            tokenizerRef.current = createQwen2ByteLevelBpeTokenizer(message.tokenizer.parts);
            tokenizerModelIdRef.current = "qwen";
            setTokenizerState("ready");
            setTokenizerError(null);
          } catch (tokenizerBuildError: unknown) {
            tokenizerRef.current = null;
            tokenizerModelIdRef.current = null;
            setTokenizerState("error");
            setTokenizerError(
              tokenizerBuildError instanceof Error
                ? tokenizerBuildError.message
                : String(tokenizerBuildError),
            );
          }
        }
        return;
      }

      if (message.type === "next-token-result") {
        if (message.performance) {
          recordGenerationPerformance(message.performance);
        }
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
      if (tokenizerState === "loading") {
        setTokenizerState("error");
      }
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
      modelId: selectedModelId,
      baseUrl: new URL(model.baseUrl, window.location.href).toString(),
      scratchSequenceLength: 256,
      ggufPath: model.ggufPath,
      ggufFallbackUrls: model.ggufFallbackUrls,
      backendPreference: model.backendPolicy.defaultPreference,
      webgpuRequired: model.backendPolicy.webgpu === "required",
    } satisfies AppWorkerRequest);
  }

  async function ensureTokenizerLoaded(modelId: ModelId) {
    const model = modelCatalog[modelId];
    if (!model.tokenizerUrl) {
      return;
    }
    if (tokenizerRef.current && tokenizerModelIdRef.current === modelId) {
      setTokenizerState("ready");
      return;
    }

    setTokenizerState("loading");
    setTokenizerError(null);
    try {
      const tokenizer = await loadByteLevelBpeTokenizer(model.tokenizerUrl, getCachedFetch());
      if (selectedModelIdRef.current !== modelId) {
        return;
      }
      tokenizerRef.current = tokenizer;
      tokenizerModelIdRef.current = modelId;
      setTokenizerState("ready");
    } catch (loadError: unknown) {
      if (selectedModelIdRef.current !== modelId) {
        return;
      }
      tokenizerRef.current = null;
      tokenizerModelIdRef.current = null;
      setTokenizerState("error");
      setTokenizerError(loadError instanceof Error ? loadError.message : String(loadError));
    }
  }

  function getCachedFetch(): typeof fetch {
    cachedFetchRef.current ??= createModelCacheFetch();
    return cachedFetchRef.current;
  }

  function selectModel(nextModelId: ModelId) {
    const nextModel = modelCatalog[nextModelId];
    if (
      nextModelId === selectedModelId ||
      loadState === "loading" ||
      generationState === "generating" ||
      !modelCanRun(nextModel, webgpuAvailability, webgpuMaxStorageBufferBindingSize)
    ) {
      return;
    }

    generationRunRef.current += 1;
    rejectPendingNextTokenRequests(
      pendingNextTokenRequestsRef.current,
      new Error("Model selection changed"),
    );
    workerRef.current?.terminate();
    workerRef.current = null;
    tokenizerRef.current = null;
    tokenizerModelIdRef.current = null;
    selectedModelIdRef.current = nextModelId;
    setSelectedModelId(nextModelId);
    setLoadState("idle");
    setTokenizerState("idle");
    setProgress(null);
    setTransientProgressMessage(null);
    setConfigRevealReady(false);
    setActualStepIndex(-1);
    setVisibleStepIndex(-1);
    setSummary(null);
    setError(null);
    setTokenizerError(null);
    setGenerationState("idle");
    setGenerationError(null);
    setGeneratedTokenIds([]);
    setInputTokenCount(null);
    resetGenerationThroughput();
  }

  async function generateCompletion() {
    const tokenizer = tokenizerRef.current;
    if (!tokenizer || loadState !== "ready" || !summary) {
      setGenerationState("error");
      setGenerationError("Model and tokenizer must be ready before generation.");
      return;
    }
    if (selectedModelId === "qwen" && qwenGenerationInFlightRef.current) {
      setGenerationState("error");
      setGenerationError("Qwen generation is still finishing its previous GPU request.");
      return;
    }

    if (chatText.length === 0) {
      setGenerationState("error");
      setGenerationError("Prompt is empty.");
      return;
    }

    const baseText = normalizeLineBreaks(chatText);
    setChatText(baseText);
    const modelPrompt = formatPromptForModel(selectedModelId, baseText);
    const inputIds = tokenizer.encode(modelPrompt);
    const availableNewTokens = summary.config.maximumSequenceLength - inputIds.length;
    const qwenSafetyError = qwenGpuBlockingPromptError(selectedModelId, summary, inputIds.length);
    if (qwenSafetyError) {
      setGenerationState("error");
      setGenerationError(qwenSafetyError);
      return;
    }
    if (availableNewTokens <= 0) {
      setGenerationState("error");
      setGenerationError("Prompt is longer than the model context window.");
      return;
    }

    const runId = generationRunRef.current + 1;
    generationRunRef.current = runId;
    const nextInputIds = [...inputIds];
    const nextGeneratedTokenIds: number[] = [];
    const targetNewTokens = Math.min(
      maxNewTokens,
      availableNewTokens,
      qwenGpuAvailableNewTokens(selectedModelId, summary, inputIds.length),
    );
    if (targetNewTokens <= 0) {
      setGenerationState("error");
      setGenerationError("Qwen WebGPU safety limit leaves no room for new tokens.");
      return;
    }
    if (selectedModelId === "qwen") {
      qwenGenerationInFlightRef.current = true;
    }

    setInputTokenCount(inputIds.length);
    setGeneratedTokenIds([]);
    resetGenerationThroughput();
    setGenerationError(null);
    setGenerationState("generating");

    try {
      for (let tokenIndex = 0; tokenIndex < targetNewTokens; tokenIndex += 1) {
        const tokenId = await requestNextToken(nextInputIds, {
          resetCache: selectedModelId === "qwen" && tokenIndex === 0,
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
        setChatText(
          normalizeLineBreaks(
            baseText +
              visibleGeneratedTextForModel(
                selectedModelId,
                tokenizer.decode(nextGeneratedTokenIds),
              ),
          ),
        );
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
    } finally {
      if (selectedModelId === "qwen") {
        qwenGenerationInFlightRef.current = false;
      }
    }
  }

  function stopGeneration() {
    generationRunRef.current += 1;
    rejectPendingNextTokenRequests(
      pendingNextTokenRequestsRef.current,
      new Error("Generation stopped"),
    );
    setGenerationState("done");
    qwenGenerationInFlightRef.current = false;
  }

  function resetGenerationThroughput() {
    decodeThroughputRef.current = { tokenCount: 0, elapsedMs: 0 };
    setGenerationThroughput({
      prefillTokensPerSecond: null,
      decodeTokensPerSecond: null,
    });
  }

  function recordGenerationPerformance(performance: AppInferencePerformance) {
    if (performance.phase === "prefill") {
      setGenerationThroughput((current) => ({
        ...current,
        prefillTokensPerSecond: performance.tokensPerSecond,
      }));
      return;
    }

    decodeThroughputRef.current = {
      tokenCount: decodeThroughputRef.current.tokenCount + performance.tokenCount,
      elapsedMs: decodeThroughputRef.current.elapsedMs + performance.elapsedMs,
    };
    setGenerationThroughput((current) => ({
      ...current,
      decodeTokensPerSecond: tokensPerSecond(
        decodeThroughputRef.current.tokenCount,
        decodeThroughputRef.current.elapsedMs,
      ),
    }));
  }

  function requestNextToken(
    inputIds: number[],
    options: { resetCache?: boolean; temperature: number; topK: number },
  ): Promise<number> {
    const worker = workerRef.current;
    if (!worker) {
      return Promise.reject(new Error("Inference worker is not running."));
    }

    const requestId = createRequestId();
    const request: AppWorkerRequest = {
      type: "next-token",
      requestId,
      modelId: selectedModelId,
      inputIds,
      resetCache: options.resetCache,
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
        selectedModelId={selectedModelId}
        modelSelectDisabled={loadState === "loading" || generationState === "generating"}
        summary={summary}
        tokenizerError={tokenizerError}
        transientProgressMessage={transientProgressMessage}
        visibleStepIndex={visibleStepIndex}
        canEnterChat={canEnterChat}
        steps={activeSteps}
        webgpuAvailability={webgpuAvailability}
        webgpuMaxStorageBufferBindingSize={webgpuMaxStorageBufferBindingSize}
        onLoadModel={loadModel}
        onModelIdChange={selectModel}
      />

      {canEnterChat && (
        <ChatSection
          canGenerate={canGenerate}
          chatText={chatText}
          decodeTokensPerSecond={generationThroughput.decodeTokensPerSecond}
          generatedTokenCount={generatedTokenIds.length}
          generationError={generationError}
          generationState={generationState}
          loadState={loadState}
          maxNewTokens={maxNewTokens}
          prefillTokensPerSecond={generationThroughput.prefillTokensPerSecond}
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

      <Footer />
    </main>
  );
}

function Footer() {
  return (
    <footer className="site-footer">
      Created by{" "}
      <a href="https://damianofds.github.io/" rel="noreferrer" target="_blank">
        fds
      </a>
    </footer>
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
  selectedModelId,
  modelSelectDisabled,
  summary,
  tokenizerError,
  transientProgressMessage,
  visibleStepIndex,
  canEnterChat,
  steps,
  onLoadModel,
  onModelIdChange,
  webgpuAvailability,
  webgpuMaxStorageBufferBindingSize,
}: {
  error: string | null;
  frame: LoadFrame;
  loadState: LoadState;
  progress: AppLoaderProgress | null;
  selectedModelId: ModelId;
  modelSelectDisabled: boolean;
  summary: AppLoadedModelSummary | null;
  tokenizerError: string | null;
  transientProgressMessage: string | null;
  visibleStepIndex: number;
  canEnterChat: boolean;
  steps: readonly AppLoadStep[];
  onLoadModel: () => void;
  onModelIdChange: (modelId: ModelId) => void;
  webgpuAvailability: WebGpuAvailability;
  webgpuMaxStorageBufferBindingSize: number | null;
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
              modelSelectDisabled={modelSelectDisabled}
              selectedModelId={selectedModelId}
              tokenizerError={tokenizerError}
              webgpuAvailability={webgpuAvailability}
              webgpuMaxStorageBufferBindingSize={webgpuMaxStorageBufferBindingSize}
              onLoadModel={onLoadModel}
              onModelIdChange={onModelIdChange}
            />
          )}
          {frame === "loading" && (
            <LoadingFrame
              loadState={loadState}
              progress={progress}
              steps={steps}
              transientProgressMessage={transientProgressMessage}
              visibleStepIndex={visibleStepIndex}
            />
          )}
          {frame === "transition" && <ConfigTransitionFrame />}
          {frame === "config" && summary && (
            <ConfigFrame
              loadState={loadState}
              modelSelectDisabled={modelSelectDisabled}
              selectedModelId={selectedModelId}
              summary={summary}
              webgpuAvailability={webgpuAvailability}
              webgpuMaxStorageBufferBindingSize={webgpuMaxStorageBufferBindingSize}
              onLoadModel={onLoadModel}
              onModelIdChange={onModelIdChange}
            />
          )}
        </div>
      </div>
      {canEnterChat && <ScrollCue href="#chat" label="Let's chat!" />}
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
  modelSelectDisabled,
  selectedModelId,
  tokenizerError,
  webgpuAvailability,
  webgpuMaxStorageBufferBindingSize,
  onLoadModel,
  onModelIdChange,
}: {
  error: string | null;
  loadState: LoadState;
  modelSelectDisabled: boolean;
  selectedModelId: ModelId;
  tokenizerError: string | null;
  webgpuAvailability: WebGpuAvailability;
  webgpuMaxStorageBufferBindingSize: number | null;
  onLoadModel: () => void;
  onModelIdChange: (modelId: ModelId) => void;
}) {
  return (
    <div className="start-load-frame">
      <button
        className="load-button"
        disabled={
          loadState === "loading" ||
          !modelCanRun(
            modelCatalog[selectedModelId],
            webgpuAvailability,
            webgpuMaxStorageBufferBindingSize,
          )
        }
        onClick={onLoadModel}
        type="button"
      >
        Start
      </button>
      <ModelSelector
        disabled={modelSelectDisabled}
        selectedModelId={selectedModelId}
        webgpuAvailability={webgpuAvailability}
        webgpuMaxStorageBufferBindingSize={webgpuMaxStorageBufferBindingSize}
        onModelIdChange={onModelIdChange}
      />
      <p className="frame-copy">
        The browser will fetch the model weights, validate them, and keep the network quiet after
        the worker is ready.
      </p>
      {error && <p className="error-message">{error}</p>}
      {tokenizerError && <p className="error-message">{tokenizerError}</p>}
    </div>
  );
}

function ModelSelector({
  disabled,
  selectedModelId,
  webgpuAvailability,
  webgpuMaxStorageBufferBindingSize,
  onModelIdChange,
}: {
  disabled: boolean;
  selectedModelId: ModelId;
  webgpuAvailability: WebGpuAvailability;
  webgpuMaxStorageBufferBindingSize: number | null;
  onModelIdChange: (modelId: ModelId) => void;
}) {
  return (
    <label className="model-selector-field">
      <span>Model</span>
      <select
        aria-label="Model"
        disabled={disabled}
        onChange={(event) => onModelIdChange(event.currentTarget.value as ModelId)}
        value={selectedModelId}
      >
        {modelOptions.map((model) => {
          const unavailable = !modelCanRun(
            model,
            webgpuAvailability,
            webgpuMaxStorageBufferBindingSize,
          );
          return (
            <option disabled={unavailable} key={model.id} value={model.id}>
              {modelDropdownLabel(
                model,
                webgpuAvailability,
                webgpuMaxStorageBufferBindingSize,
              )}
            </option>
          );
        })}
      </select>
    </label>
  );
}

function LoadingFrame({
  loadState,
  progress,
  steps,
  transientProgressMessage,
  visibleStepIndex,
}: {
  loadState: LoadState;
  progress: AppLoaderProgress | null;
  steps: readonly AppLoadStep[];
  transientProgressMessage: string | null;
  visibleStepIndex: number;
}) {
  return (
    <div className="loading-frame-inner">
      <div className="progress-track" aria-label="Model loading progress">
        <div className="progress-fill" style={{ width: `${stepPercent(visibleStepIndex, steps)}%` }} />
      </div>

      <p className="current-message">
        {transientProgressMessage ?? currentStepMessage(visibleStepIndex, loadState, steps)}
        {!transientProgressMessage && progress?.loadedBytes && progress.totalBytes
          ? ` / ${formatBytes(progress.loadedBytes)} of ${formatBytes(progress.totalBytes)}`
          : ""}
      </p>

      <ol className="step-list">
        {steps.map((step, index) => (
          <li className={stepClassName(index, visibleStepIndex)} key={step.key}>
            <span className="step-index">{String(index + 1).padStart(2, "0")}</span>
            <span>{step.label}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ConfigTransitionFrame() {
  return (
    <div className="config-transition-frame" aria-live="polite">
      <div className="config-transition-grid" aria-hidden="true">
        {transitionTiles.map((tile) => (
          <span key={tile} style={{ animationDelay: `${tile * 35}ms` }} />
        ))}
      </div>
      <p>Preparing model and tensor config</p>
    </div>
  );
}

function ConfigFrame({
  loadState,
  modelSelectDisabled,
  selectedModelId,
  summary,
  webgpuAvailability,
  webgpuMaxStorageBufferBindingSize,
  onLoadModel,
  onModelIdChange,
}: {
  loadState: LoadState;
  modelSelectDisabled: boolean;
  selectedModelId: ModelId;
  summary: AppLoadedModelSummary;
  webgpuAvailability: WebGpuAvailability;
  webgpuMaxStorageBufferBindingSize: number | null;
  onLoadModel: () => void;
  onModelIdChange: (modelId: ModelId) => void;
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
        <LayerStrip summary={summary} />
      </section>

      <section className="config-panel">
        <h3>Tensor config</h3>
        <div className="config-grid tensor-config-grid">
          {tensorConfigRows(summary).map((row) => (
            <ConfigItem key={row.label} label={row.label} value={row.value} />
          ))}
        </div>
      </section>

      <div className="config-actions">
        <button className="reload-link" onClick={onLoadModel} type="button">
          Reload model
        </button>
        <ModelSelector
          disabled={modelSelectDisabled}
          selectedModelId={selectedModelId}
          webgpuAvailability={webgpuAvailability}
          webgpuMaxStorageBufferBindingSize={webgpuMaxStorageBufferBindingSize}
          onModelIdChange={onModelIdChange}
        />
      </div>
    </div>
  );
}

function LayerStrip({ summary }: { summary: AppLoadedModelSummary }) {
  if (summary.kind === "gpt-neo") {
    return (
      <div className="attention-strip" aria-label="Attention layer types">
        {summary.config.attentionLayers.map((kind, index) => (
          <span className={kind} key={`${kind}-${index}`} title={`Layer ${index}: ${kind}`}>
            {index}
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="attention-strip" aria-label="Qwen GQA layers">
      {Array.from({ length: summary.layers }, (_, index) => (
        <span className="qwen" key={index} title={`Layer ${index}: grouped query attention`}>
          {index}
        </span>
      ))}
    </div>
  );
}

function ChatSection({
  canGenerate,
  chatText,
  decodeTokensPerSecond,
  generatedTokenCount,
  generationError,
  generationState,
  loadState,
  maxNewTokens,
  prefillTokensPerSecond,
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
  decodeTokensPerSecond: number | null;
  generatedTokenCount: number;
  generationError: string | null;
  generationState: GenerationState;
  loadState: LoadState;
  maxNewTokens: number;
  prefillTokensPerSecond: number | null;
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
          <h2>Chat with broSLM</h2>
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
              <span>Prefill {formatTokensPerSecond(prefillTokensPerSecond)}</span>
              <span>Decode {formatTokensPerSecond(decodeTokensPerSecond)}</span>
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
  summary: AppLoadedModelSummary | null,
  visibleStepIndex: number,
  configRevealReady: boolean,
  steps: readonly AppLoadStep[],
): LoadFrame {
  if (summary && visibleStepIndex >= steps.length - 1) {
    return configRevealReady ? "config" : "transition";
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
  if (frame === "transition") {
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

function qwenGpuBlockingPromptError(
  selectedModelId: ModelId,
  summary: AppLoadedModelSummary | null,
  tokenCount: number | null,
): string | null {
  if (
    selectedModelId !== "qwen" ||
    summary?.kind !== "qwen2" ||
    summary.backend !== "webgpu" ||
    tokenCount === null
  ) {
    return null;
  }

  const prefillError = qwen2WebGpuPrefillSafetyError(tokenCount);
  if (prefillError) {
    return prefillError;
  }
  if (tokenCount >= qwen2WebGpuSafetyLimits.maxSequenceTokens) {
    return (
      `Qwen WebGPU cache is capped at ${qwen2WebGpuSafetyLimits.maxSequenceTokens} ` +
      `tokens for GPU stability. Current prompt is ${tokenCount} tokens.`
    );
  }
  return null;
}

function qwenGpuAvailableNewTokens(
  selectedModelId: ModelId,
  summary: AppLoadedModelSummary,
  promptTokenCount: number,
): number {
  if (selectedModelId !== "qwen" || summary.kind !== "qwen2" || summary.backend !== "webgpu") {
    return Number.MAX_SAFE_INTEGER;
  }
  return Math.max(0, qwen2WebGpuSafetyLimits.maxSequenceTokens - promptTokenCount);
}

function tokensPerSecond(tokenCount: number, elapsedMs: number): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    return 0;
  }
  return tokenCount / (elapsedMs / 1000);
}

function formatTokensPerSecond(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value <= 0) {
    return "- tok/s";
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} tok/s`;
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

function modelCanRun(
  model: ModelCatalogEntry,
  webgpuAvailability: WebGpuAvailability,
  webgpuMaxStorageBufferBindingSize: number | null,
): boolean {
  if (model.backendPolicy.cpuFallback) {
    return true;
  }

  return (
    webgpuAvailability === "available" &&
    modelWebGpuRequirementsMet(model, webgpuMaxStorageBufferBindingSize)
  );
}

function modelDropdownLabel(
  model: ModelCatalogEntry,
  webgpuAvailability: WebGpuAvailability,
  webgpuMaxStorageBufferBindingSize: number | null,
): string {
  const runtimeLabel = modelRuntimeLabel(
    model,
    webgpuAvailability,
    webgpuMaxStorageBufferBindingSize,
  );
  return `${model.label} - ${runtimeLabel}`;
}

function modelRuntimeLabel(
  model: ModelCatalogEntry,
  webgpuAvailability: WebGpuAvailability,
  webgpuMaxStorageBufferBindingSize: number | null,
): string {
  if (webgpuAvailability === "checking") {
    return model.backendPolicy.cpuFallback ? "checking GPU, CPU fallback" : "checking GPU";
  }

  if (webgpuAvailability === "available") {
    if (model.backendPolicy.webgpu === "unsupported") {
      return "CPU only";
    }
    if (!modelWebGpuRequirementsMet(model, webgpuMaxStorageBufferBindingSize)) {
      return "GPU limit too low";
    }
    return model.backendPolicy.cpuFallback ? "GPU available, CPU fallback" : "GPU available";
  }

  if (model.backendPolicy.cpuFallback) {
    return "GPU unavailable, CPU fallback";
  }

  return "GPU unavailable";
}

function modelUnavailableMessage(
  model: ModelCatalogEntry,
  webgpuAvailability: WebGpuAvailability,
  webgpuMaxStorageBufferBindingSize: number | null,
): string {
  if (webgpuAvailability === "checking") {
    return `${model.label} requires WebGPU. WebGPU support is still being checked.`;
  }
  if (
    webgpuAvailability === "available" &&
    !modelWebGpuRequirementsMet(model, webgpuMaxStorageBufferBindingSize)
  ) {
    return (
      `${model.label} requires a WebGPU storage buffer binding size of at least ` +
      `${formatBytes(model.backendPolicy.minimumStorageBufferBindingSize ?? 0)}, but this ` +
      `adapter reports ${formatBytes(webgpuMaxStorageBufferBindingSize ?? 0)}.`
    );
  }

  return `${model.label} requires WebGPU, but WebGPU is not available in this browser.`;
}

function modelWebGpuRequirementsMet(
  model: ModelCatalogEntry,
  webgpuMaxStorageBufferBindingSize: number | null,
): boolean {
  const requiredStorageBindingSize = model.backendPolicy.minimumStorageBufferBindingSize;
  if (!requiredStorageBindingSize) {
    return true;
  }
  return (
    webgpuMaxStorageBufferBindingSize !== null &&
    webgpuMaxStorageBufferBindingSize >= requiredStorageBindingSize
  );
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

function stepClassName(index: number, visibleStepIndex: number): string {
  if (index < visibleStepIndex) {
    return "done";
  }
  if (index === visibleStepIndex) {
    return "active";
  }
  return "";
}

function stepPercent(visibleStepIndex: number, steps: readonly AppLoadStep[]): number {
  if (visibleStepIndex < 0) {
    return 0;
  }
  return Math.min(100, ((visibleStepIndex + 1) / steps.length) * 100);
}

function currentStepMessage(
  visibleStepIndex: number,
  loadState: LoadState,
  steps: readonly AppLoadStep[],
): string {
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

function modelConfigRows(summary: AppLoadedModelSummary): Array<{ label: string; value: string }> {
  if (summary.kind === "qwen2") {
    const { config } = summary;
    return [
      { label: "Model", value: summary.modelLabel },
      { label: "Backend", value: summary.backend.toUpperCase() },
      { label: "Architecture", value: summary.architecture },
      { label: "Dtype", value: summary.dtype },
      { label: "Layers", value: formatInteger(summary.layers) },
      { label: "Hidden size", value: formatInteger(summary.hiddenSize) },
      { label: "Vocabulary", value: formatInteger(summary.vocabularySize) },
      { label: "Max sequence", value: formatInteger(config.maximumSequenceLength) },
      { label: "Intermediate", value: formatInteger(config.intermediateSize) },
      { label: "Query heads", value: formatInteger(config.numberOfHeads) },
      { label: "KV heads", value: formatInteger(config.numberOfKeyValueHeads) },
      { label: "Head dim", value: formatInteger(config.headDimension) },
      { label: "KV hidden", value: formatInteger(config.keyValueHiddenSize) },
      { label: "RoPE theta", value: formatInteger(config.ropeTheta) },
      { label: "RMS eps", value: String(config.rmsNormEpsilon) },
      { label: "BOS / EOS / PAD", value: `${config.bosTokenId} / ${config.eosTokenId} / ${config.padTokenId ?? "null"}` },
      { label: "Tied embeddings", value: config.tiedWordEmbeddings ? "yes" : "no" },
    ];
  }

  const { config } = summary;
  return [
    { label: "Model", value: summary.modelLabel },
    { label: "Backend", value: summary.backend.toUpperCase() },
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

function tensorConfigRows(summary: AppLoadedModelSummary): Array<{ label: string; value: string }> {
  const tokenEmbedding =
    summary.kind === "qwen2"
      ? tensorByName(summary, "token_embd.weight")
      : tensorByName(summary, "transformer.wte.weight");
  const positionEmbedding =
    summary.kind === "qwen2" ? null : tensorByName(summary, "transformer.wpe.weight");
  const lmHead =
    summary.kind === "qwen2" ? tensorByName(summary, "output.weight") : tensorByName(summary, "lm_head.weight");
  const finalNorm =
    summary.kind === "qwen2"
      ? tensorByName(summary, "output_norm.weight")
      : tensorByName(summary, "transformer.ln_f.weight");

  const rows = [
    { label: "Tensor count", value: formatInteger(summary.tensorCount) },
    { label: "Weight bytes", value: formatBytes(summary.totalByteLength) },
    { label: "Token embedding", value: tokenEmbedding ? formatShape(tokenEmbedding.shape) : "-" },
    { label: "LM head", value: lmHead ? formatShape(lmHead.shape) : "-" },
    { label: "Final norm", value: finalNorm ? formatShape(finalNorm.shape) : "-" },
    { label: "Largest tensor", value: largestTensorLabel(summary) },
  ];

  if (summary.kind === "gpt-neo") {
    rows.splice(2, 0, { label: "Scratch sequence", value: formatInteger(summary.scratchSequenceLength) });
    rows.splice(4, 0, {
      label: "Position embedding",
      value: positionEmbedding ? formatShape(positionEmbedding.shape) : "-",
    });
  } else {
    rows.splice(2, 0, { label: "KV hidden", value: formatInteger(summary.keyValueHiddenSize) });
  }

  return rows;
}

function tensorByName(summary: AppLoadedModelSummary, name: string) {
  return summary.tensors.find((tensor) => tensor.name === name) ?? null;
}

function largestTensorLabel(summary: AppLoadedModelSummary): string {
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

import { useMemo, useRef, useState } from "react";
import type {
  LoadedModelSummary,
  LoaderProgress,
  LoaderWorkerRequest,
  LoaderWorkerResponse,
  ModelConfig,
  TensorVisualization,
} from "../engine/src/loader";
import { modelBaseUrl } from "./modelExport";

type LoadState = "idle" | "loading" | "ready" | "error";

interface Step {
  stage: LoaderProgress["stage"];
  label: string;
}

const steps: Step[] = [
  { stage: "descriptors-download-started", label: "Start descriptor downloads" },
  { stage: "descriptors-downloaded", label: "Download config.json and weights.json" },
  { stage: "descriptors-validated", label: "Validate architecture, dtype, tensor names and sizes" },
  { stage: "weights-download-started", label: "Download weights.bin into one ArrayBuffer" },
  { stage: "weights-validated", label: "Validate binary length and tensor boundaries" },
  { stage: "tensor-views-created", label: "Create zero-copy Float32Array tensor views" },
  { stage: "weights-bound", label: "Bind raw tensors into typed GPT-Neo layers" },
  { stage: "scratch-allocated", label: "Allocate runtime scratch buffers separately" },
  { stage: "ready", label: "Keep model alive in the worker and notify the page" },
];

export default function App() {
  const workerRef = useRef<Worker | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [progress, setProgress] = useState<LoaderProgress | null>(null);
  const [completedStages, setCompletedStages] = useState<Set<LoaderProgress["stage"]>>(
    () => new Set(),
  );
  const [summary, setSummary] = useState<LoadedModelSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tensorFilter, setTensorFilter] = useState("");
  const [selectedTensorName, setSelectedTensorName] = useState<string | null>(null);

  const percent = useMemo(() => {
    if (!progress?.loadedBytes || !progress.totalBytes) {
      return null;
    }
    return Math.max(0, Math.min(100, (progress.loadedBytes / progress.totalBytes) * 100));
  }, [progress]);

  const filteredTensors = useMemo(() => {
    const normalizedFilter = tensorFilter.trim().toLowerCase();
    const tensors = summary?.tensors ?? [];
    if (!normalizedFilter) {
      return tensors;
    }

    return tensors.filter((tensor) => tensor.name.toLowerCase().includes(normalizedFilter));
  }, [summary?.tensors, tensorFilter]);

  const selectedTensor = useMemo(() => {
    return (
      filteredTensors.find((tensor) => tensor.name === selectedTensorName) ??
      filteredTensors[0] ??
      null
    );
  }, [filteredTensors, selectedTensorName]);

  function loadModel() {
    if (loadState === "loading") {
      return;
    }

    setLoadState("loading");
    setProgress(null);
    setCompletedStages(new Set());
    setSummary(null);
    setError(null);

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
        setCompletedStages((current) => {
          const next = new Set(current);
          next.add(message.progress.stage);
          return next;
        });
        return;
      }

      if (message.type === "model-ready") {
        setSummary(message.summary);
        setSelectedTensorName(message.summary.tensors[0]?.name ?? null);
        setLoadState("ready");
        return;
      }

      if (message.type === "next-token-result") {
        return;
      }

      setError(message.error);
      setLoadState("error");
    };

    worker.onerror = (event) => {
      setError(event.message || "The inference worker failed while loading the model.");
      setLoadState("error");
    };

    const request: LoaderWorkerRequest = {
      type: "load-model",
      requestId: crypto.randomUUID(),
      baseUrl: new URL(modelBaseUrl, window.location.href).toString(),
      scratchSequenceLength: 256,
    };
    worker.postMessage(request);
  }

  return (
    <main className="page-shell">
      <section className="intro-band">
        <div className="intro-copy">
          <p className="eyebrow">TinyStories GPT-Neo export</p>
          <h1>broSLM</h1>
          <p className="subtitle">browser small language model</p>
          <p className="description">
            This page loads the TinyStories raw binary model in a dedicated inference Web
            Worker. The UI stays on the main thread while the worker validates descriptors,
            downloads weights, creates zero-copy tensor views, and keeps the model resident.
          </p>
          <button className="load-button" onClick={loadModel} disabled={loadState === "loading"}>
            {loadState === "loading" ? "Loading model" : loadState === "ready" ? "Reload model" : "Load model"}
          </button>
        </div>
      </section>

      <section className="status-band" aria-live="polite">
        <div className="status-header">
          <div>
            <p className="section-label">Loader state</p>
            <h2>{statusTitle(loadState)}</h2>
          </div>
          {percent !== null && (
            <span className="download-pill">{formatBytes(progress?.loadedBytes ?? 0)} / {formatBytes(progress?.totalBytes ?? 0)}</span>
          )}
        </div>

        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${percent ?? stagePercent(completedStages)}%` }}
          />
        </div>

        {progress && <p className="current-message">{progress.message}</p>}
        {error && <p className="error-message">{error}</p>}

        <ol className="step-list">
          {steps.map((step) => {
            const done = isStepDone(step.stage, completedStages);
            const active = isStepActive(step.stage, progress?.stage, loadState);
            return (
              <li className={done ? "done" : active ? "active" : ""} key={step.stage}>
                <span className="step-marker" />
                <span>{step.label}</span>
              </li>
            );
          })}
        </ol>

        {summary && (
          <>
            <div className="summary-grid">
              <SummaryItem label="Architecture" value={summary.architecture} />
              <SummaryItem label="Tensors" value={String(summary.tensorCount)} />
              <SummaryItem label="Weights" value={formatBytes(summary.totalByteLength)} />
              <SummaryItem label="Layers" value={String(summary.layers)} />
              <SummaryItem label="Hidden size" value={String(summary.hiddenSize)} />
              <SummaryItem label="Vocabulary" value={String(summary.vocabularySize)} />
            </div>

            <ModelDetails
              config={summary.config}
              tensors={filteredTensors}
              totalTensors={summary.tensors.length}
              selectedTensor={selectedTensor}
              tensorFilter={tensorFilter}
              onTensorFilterChange={setTensorFilter}
              onSelectTensor={setSelectedTensorName}
            />
          </>
        )}
      </section>
    </main>
  );
}

function ModelDetails({
  config,
  tensors,
  totalTensors,
  selectedTensor,
  tensorFilter,
  onTensorFilterChange,
  onSelectTensor,
}: {
  config: ModelConfig;
  tensors: TensorVisualization[];
  totalTensors: number;
  selectedTensor: TensorVisualization | null;
  tensorFilter: string;
  onTensorFilterChange: (value: string) => void;
  onSelectTensor: (name: string) => void;
}) {
  return (
    <div className="details-layout">
      <section className="detail-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">Model config</p>
            <h3>Parsed architecture</h3>
          </div>
        </div>
        <div className="config-grid">
          {configRows(config).map((row) => (
            <div className="config-row" key={row.label}>
              <span>{row.label}</span>
              <strong>{row.value}</strong>
            </div>
          ))}
        </div>
        <div className="attention-strip" aria-label="Attention layer types">
          {config.attentionLayers.map((kind, index) => (
            <span className={kind} key={`${kind}-${index}`} title={`Layer ${index}: ${kind}`}>
              {index}
            </span>
          ))}
        </div>
      </section>

      <section className="detail-panel tensor-panel">
        <div className="panel-heading tensor-heading">
          <div>
            <p className="section-label">Tensor visualization</p>
            <h3>Loaded tensor registry</h3>
          </div>
          <label className="tensor-filter">
            <span>Filter</span>
            <input
              type="search"
              value={tensorFilter}
              onChange={(event) => onTensorFilterChange(event.target.value)}
              placeholder="tensor name"
            />
          </label>
        </div>

        <p className="tensor-count">
          Showing {tensors.length} of {totalTensors} tensors read from the loaded model.
        </p>

        <div className="tensor-browser">
          <div className="tensor-list" role="listbox" aria-label="Loaded tensors">
            {tensors.map((tensor) => (
              <button
                className={tensor.name === selectedTensor?.name ? "tensor-row selected" : "tensor-row"}
                key={tensor.name}
                onClick={() => onSelectTensor(tensor.name)}
                type="button"
              >
                <span className="tensor-name">{tensor.name}</span>
                <span className="tensor-shape">{formatShape(tensor.shape)}</span>
                <span className="tensor-size">{formatBytes(tensor.byteLength)}</span>
              </button>
            ))}
          </div>

          {selectedTensor && <TensorInspector tensor={selectedTensor} />}
        </div>
      </section>
    </div>
  );
}

function TensorInspector({ tensor }: { tensor: TensorVisualization }) {
  return (
    <div className="tensor-inspector">
      <div>
        <p className="selected-label">Selected tensor</p>
        <h4>{tensor.name}</h4>
        <p className="tensor-description">{tensor.description}</p>
      </div>
      <div className="tensor-metrics">
        <SummaryItem label="Shape" value={formatShape(tensor.shape)} />
        <SummaryItem label="Elements" value={formatInteger(tensor.elementCount)} />
        <SummaryItem label="Offset" value={formatBytes(tensor.byteOffset)} />
        <SummaryItem label="Mean abs" value={formatFloat(tensor.meanAbsolute)} />
        <SummaryItem label="Min" value={formatFloat(tensor.min)} />
        <SummaryItem label="Max" value={formatFloat(tensor.max)} />
      </div>
      <div className="sample-bars" aria-label="Sampled tensor values">
        {tensor.sample.map((value, index) => (
          <span
            className={value >= 0 ? "positive" : "negative"}
            key={`${index}-${value}`}
            style={{ height: `${sampleBarHeight(value, tensor)}%` }}
            title={formatFloat(value)}
          />
        ))}
      </div>
    </div>
  );
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function statusTitle(loadState: LoadState): string {
  switch (loadState) {
    case "loading":
      return "Loading inside the inference worker";
    case "ready":
      return "Model ready";
    case "error":
      return "Load failed";
    default:
      return "Waiting to load";
  }
}

function stagePercent(completedStages: Set<LoaderProgress["stage"]>): number {
  const visibleCompletedSteps = steps.filter((step) => isStepDone(step.stage, completedStages)).length;
  if (visibleCompletedSteps === 0) {
    return 0;
  }
  return Math.min(100, (visibleCompletedSteps / steps.length) * 100);
}

function isStepDone(
  stage: LoaderProgress["stage"],
  completedStages: Set<LoaderProgress["stage"]>,
): boolean {
  if (stage === "weights-download-started") {
    return (
      completedStages.has("weights-download-started") ||
      completedStages.has("weights-download-progress") ||
      completedStages.has("weights-downloaded")
    );
  }

  return completedStages.has(stage);
}

function isStepActive(
  stage: LoaderProgress["stage"],
  currentStage: LoaderProgress["stage"] | undefined,
  loadState: LoadState,
): boolean {
  if (loadState !== "loading") {
    return false;
  }
  if (stage === "weights-download-started") {
    return currentStage === "weights-download-started" || currentStage === "weights-download-progress";
  }

  return currentStage === stage;
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

function configRows(config: ModelConfig): Array<{ label: string; value: string }> {
  return [
    { label: "Architecture", value: config.architecture },
    { label: "Activation", value: config.activation },
    { label: "Vocabulary", value: formatInteger(config.vocabularySize) },
    { label: "Hidden size", value: formatInteger(config.hiddenSize) },
    { label: "Intermediate", value: formatInteger(config.intermediateSize) },
    { label: "Layers", value: formatInteger(config.numberOfLayers) },
    { label: "Heads", value: formatInteger(config.numberOfHeads) },
    { label: "Head dim", value: formatInteger(config.headDimension) },
    { label: "Max sequence", value: formatInteger(config.maximumSequenceLength) },
    { label: "Window size", value: formatInteger(config.windowSize) },
    { label: "Layer norm eps", value: String(config.layerNormEpsilon) },
    { label: "BOS / EOS / PAD", value: `${config.bosTokenId} / ${config.eosTokenId} / ${config.padTokenId ?? "null"}` },
    { label: "Tied embeddings", value: config.tiedWordEmbeddings ? "yes" : "no" },
  ];
}

function formatShape(shape: readonly number[]): string {
  return `[${shape.map(formatInteger).join(", ")}]`;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatFloat(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(3);
  }
  return value.toFixed(5);
}

function sampleBarHeight(value: number, tensor: TensorVisualization): number {
  const scale = Math.max(Math.abs(tensor.min), Math.abs(tensor.max), 1e-8);
  return Math.max(4, Math.min(100, (Math.abs(value) / scale) * 100));
}

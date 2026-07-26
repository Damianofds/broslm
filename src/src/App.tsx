import { useMemo, useRef, useState } from "react";
import type {
  LoadedModelSummary,
  LoaderProgress,
  LoaderWorkerRequest,
  LoaderWorkerResponse,
} from "../engine/loader";
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

  const percent = useMemo(() => {
    if (!progress?.loadedBytes || !progress.totalBytes) {
      return null;
    }
    return Math.max(0, Math.min(100, (progress.loadedBytes / progress.totalBytes) * 100));
  }, [progress]);

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
        setLoadState("ready");
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
          <div className="summary-grid">
            <SummaryItem label="Architecture" value={summary.architecture} />
            <SummaryItem label="Tensors" value={String(summary.tensorCount)} />
            <SummaryItem label="Weights" value={formatBytes(summary.totalByteLength)} />
            <SummaryItem label="Layers" value={String(summary.layers)} />
            <SummaryItem label="Hidden size" value={String(summary.hiddenSize)} />
            <SummaryItem label="Vocabulary" value={String(summary.vocabularySize)} />
          </div>
        )}
      </section>
    </main>
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

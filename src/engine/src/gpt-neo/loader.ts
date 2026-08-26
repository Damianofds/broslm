import { allocateModelKvCache, type ModelKvCache } from "./attentionCache";
import { nextTokenWithCache } from "./model";
import { isModelAssetCacheHit, type ModelAssetFetchSource } from "../modelAssetFetch";
import type { TensorDescriptor, TensorView, WeightsIndex } from "../tensor";

export type { TensorDescriptor, TensorView, WeightsIndex } from "../tensor";

export type AttentionKind = "global" | "local";

export interface ModelConfig {
  architecture: "gpt_neo";
  vocabularySize: number;
  hiddenSize: number;
  intermediateSize: number;
  numberOfLayers: number;
  numberOfHeads: number;
  headDimension: number;
  maximumSequenceLength: number;
  layerNormEpsilon: number;
  activation: "gelu_new";
  tiedWordEmbeddings: boolean;
  attentionLayers: AttentionKind[];
  attentionTypes?: unknown;
  windowSize: number;
  bosTokenId: number;
  eosTokenId: number;
  padTokenId: number | null;
}

export interface LayerNormWeights {
  weight: TensorView;
  bias: TensorView;
}

export interface AttentionWeights {
  kind: AttentionKind;
  kProjWeight: TensorView;
  vProjWeight: TensorView;
  qProjWeight: TensorView;
  outProjWeight: TensorView;
  outProjBias: TensorView;
}

export interface MlpWeights {
  cFcWeight: TensorView;
  cFcBias: TensorView;
  cProjWeight: TensorView;
  cProjBias: TensorView;
}

export interface TransformerLayerWeights {
  index: number;
  ln1: LayerNormWeights;
  attention: AttentionWeights;
  ln2: LayerNormWeights;
  mlp: MlpWeights;
}

export interface BoundModelWeights {
  tokenEmbedding: TensorView;
  positionEmbedding: TensorView;
  layers: TransformerLayerWeights[];
  finalLayerNorm: LayerNormWeights;
  lmHead: TensorView;
}

export interface RuntimeScratch {
  sequenceLength: number;
  hiddenState: Float32Array;
  residual: Float32Array;
  normed: Float32Array;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  attentionOutput: Float32Array;
  attentionScores: Float32Array;
  mlpIntermediate: Float32Array;
  logits: Float32Array;
}

export interface LoadedModel {
  config: ModelConfig;
  weightsIndex: WeightsIndex;
  weightsBuffer: ArrayBuffer;
  tensors: ReadonlyMap<string, TensorView>;
  weights: BoundModelWeights;
  scratch: RuntimeScratch;
}

export interface LoadModelOptions {
  baseUrl: string;
  configPath?: string;
  weightsIndexPath?: string;
  weightsBinaryPath?: string;
  scratchSequenceLength?: number;
  fetchImpl?: typeof fetch;
  onProgress?: (progress: LoaderProgress) => void;
}

export type LoaderWorkerRequest =
  | {
      type: "load-model";
      requestId?: string;
      baseUrl: string;
      configPath?: string;
      weightsIndexPath?: string;
      weightsBinaryPath?: string;
      scratchSequenceLength?: number;
    }
  | {
      type: "next-token";
      requestId?: string;
      inputIds: number[];
      temperature?: number;
      topK?: number;
    };

export type LoaderWorkerResponse =
  | {
      type: "model-progress";
      requestId?: string;
      progress: LoaderProgress;
    }
  | {
      type: "model-ready";
      requestId?: string;
      summary: LoadedModelSummary;
    }
  | {
      type: "next-token-result";
      requestId?: string;
      tokenId: number;
    }
  | {
      type: "model-error";
      requestId?: string;
      error: string;
    };

export interface LoadedModelSummary {
  architecture: string;
  dtype: string;
  tensorCount: number;
  totalByteLength: number;
  layers: number;
  hiddenSize: number;
  vocabularySize: number;
  maximumSequenceLength: number;
  scratchSequenceLength: number;
  config: ModelConfig;
  tensors: TensorVisualization[];
}

export interface TensorVisualization {
  name: string;
  description: string;
  shape: readonly number[];
  byteOffset: number;
  byteLength: number;
  elementCount: number;
  min: number;
  max: number;
  mean: number;
  meanAbsolute: number;
  sample: number[];
}

export interface LoaderProgress {
  stage:
    | "descriptors-download-started"
    | "descriptors-downloaded"
    | "descriptors-validated"
    | "weights-download-started"
    | "weights-download-progress"
    | "weights-downloaded"
    | "weights-validated"
    | "tensor-views-created"
    | "weights-bound"
    | "scratch-allocated"
    | "ready";
  message: string;
  source?: "network" | "cache";
  loadedBytes?: number;
  totalBytes?: number;
}

interface LoaderWorkerScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<LoaderWorkerRequest>) => void,
  ): void;
  postMessage(message: LoaderWorkerResponse): void;
}

const FLOAT32_BYTES = 4;

let workerLoadedModel: LoadedModel | null = null;
let workerKvCache: ModelKvCache | null = null;

export async function loadModel(options: LoadModelOptions): Promise<LoadedModel> {
  const fetcher = options.fetchImpl ?? fetch;
  const report = options.onProgress ?? (() => undefined);
  const configUrl = resolveModelUrl(options.baseUrl, options.configPath ?? "config.json");
  const weightsIndexUrl = resolveModelUrl(
    options.baseUrl,
    options.weightsIndexPath ?? "weights.json",
  );

  report({
    stage: "descriptors-download-started",
    message: "Downloading config.json and weights.json",
  });
  const [config, weightsIndex] = await Promise.all([
    fetchJson<ModelConfig>(fetcher, configUrl),
    fetchJson<WeightsIndex>(fetcher, weightsIndexUrl),
  ]);

  report({
    stage: "descriptors-downloaded",
    message: "Downloaded model descriptors",
  });
  validateConfig(config);
  validateWeightsIndex(config, weightsIndex);
  report({
    stage: "descriptors-validated",
    message: `Validated ${weightsIndex.tensorCount} tensor descriptors`,
    totalBytes: weightsIndex.totalByteLength,
  });

  const weightsBinaryUrl = resolveModelUrl(
    options.baseUrl,
    options.weightsBinaryPath ?? "weights.bin",
  );
  report({
    stage: "weights-download-started",
    message: "Downloading weights.bin",
    totalBytes: weightsIndex.totalByteLength,
  });
  const weightsBuffer = await fetchArrayBuffer(
    fetcher,
    weightsBinaryUrl,
    (loadedBytes, totalBytes, source) => {
      report({
        stage: "weights-download-progress",
        message:
          source === "cache" ? "Model weights found in browser cache" : "Downloading weights.bin",
        source,
        loadedBytes,
        totalBytes: totalBytes ?? weightsIndex.totalByteLength,
      });
    },
  );

  report({
    stage: "weights-downloaded",
    message: "Downloaded weights.bin",
    loadedBytes: weightsBuffer.byteLength,
    totalBytes: weightsIndex.totalByteLength,
  });
  validateWeightsBuffer(weightsBuffer, weightsIndex);
  report({
    stage: "weights-validated",
    message: "Validated weights.bin length and tensor boundaries",
    loadedBytes: weightsBuffer.byteLength,
    totalBytes: weightsIndex.totalByteLength,
  });
  const tensors = createTensorViews(weightsBuffer, weightsIndex);
  report({
    stage: "tensor-views-created",
    message: "Created zero-copy Float32Array tensor views",
  });
  const weights = bindModelWeights(config, tensors);
  report({
    stage: "weights-bound",
    message: "Bound raw tensor names into typed GPT-Neo layers",
  });
  const scratch = allocateRuntimeScratch(
    config,
    options.scratchSequenceLength ?? Math.min(config.maximumSequenceLength, 256),
  );
  report({
    stage: "scratch-allocated",
    message: "Allocated runtime scratch buffers",
  });

  const loadedModel = {
    config,
    weightsIndex,
    weightsBuffer,
    tensors,
    weights,
    scratch,
  };

  report({
    stage: "ready",
    message: "Model is ready inside the inference worker",
  });

  return loadedModel;
}

export function installLoaderWorker(
  selfScope: LoaderWorkerScope = globalThis as unknown as LoaderWorkerScope,
  fetchImpl?: typeof fetch,
): void {
  selfScope.addEventListener("message", (event: MessageEvent<LoaderWorkerRequest>) => {
    const message = event.data;
    if (!message) {
      return;
    }

    if (message.type === "next-token") {
      try {
        if (!workerLoadedModel) {
          throw new Error("Model must be loaded before running next-token inference");
        }

        if (!workerKvCache) {
          workerKvCache = allocateModelKvCache(workerLoadedModel.config);
        }

        const result = nextTokenWithCache(workerLoadedModel, message.inputIds, workerKvCache, {
          temperature: message.temperature,
          topK: message.topK,
        });
        selfScope.postMessage({
          type: "next-token-result",
          requestId: message.requestId,
          tokenId: result.tokenId,
        } satisfies LoaderWorkerResponse);
      } catch (error: unknown) {
        selfScope.postMessage({
          type: "model-error",
          requestId: message.requestId,
          error: error instanceof Error ? error.message : String(error),
        } satisfies LoaderWorkerResponse);
      }
      return;
    }

    if (message.type !== "load-model") {
      return;
    }

    workerLoadedModel = null;
    workerKvCache = null;

    void loadModel({
      ...message,
      fetchImpl,
      onProgress: (progress) => {
        selfScope.postMessage({
          type: "model-progress",
          requestId: message.requestId,
          progress,
        });
      },
    })
      .then((loadedModel) => {
        workerLoadedModel = loadedModel;
        workerKvCache = allocateModelKvCache(loadedModel.config);
        selfScope.postMessage({
          type: "model-ready",
          requestId: message.requestId,
          summary: summarizeLoadedModel(loadedModel),
        } satisfies LoaderWorkerResponse);
      })
      .catch((error: unknown) => {
        workerLoadedModel = null;
        workerKvCache = null;
        selfScope.postMessage({
          type: "model-error",
          requestId: message.requestId,
          error: error instanceof Error ? error.message : String(error),
        } satisfies LoaderWorkerResponse);
      });
  });
}

export function getWorkerLoadedModel(): LoadedModel | null {
  return workerLoadedModel;
}

export function summarizeLoadedModel(model: LoadedModel): LoadedModelSummary {
  return {
    architecture: model.config.architecture,
    dtype: model.weightsIndex.dtype,
    tensorCount: model.weightsIndex.tensorCount,
    totalByteLength: model.weightsIndex.totalByteLength,
    layers: model.config.numberOfLayers,
    hiddenSize: model.config.hiddenSize,
    vocabularySize: model.config.vocabularySize,
    maximumSequenceLength: model.config.maximumSequenceLength,
    scratchSequenceLength: model.scratch.sequenceLength,
    config: model.config,
    tensors: summarizeTensors(model.tensors),
  };
}

export function summarizeTensors(tensors: ReadonlyMap<string, TensorView>): TensorVisualization[] {
  return [...tensors.values()].map((tensor) => summarizeTensor(tensor));
}

function summarizeTensor(tensor: TensorView): TensorVisualization {
  const sampleLimit = 96;
  const statsLimit = 4096;
  const sample: number[] = [];
  const data = tensor.data;
  const sampleStride = Math.max(1, Math.floor(data.length / sampleLimit));
  const statsStride = Math.max(1, Math.floor(data.length / statsLimit));

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let absoluteSum = 0;
  let count = 0;

  for (let index = 0; index < data.length; index += statsStride) {
    const value = data[index] ?? 0;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    absoluteSum += Math.abs(value);
    count += 1;
  }

  for (let index = 0; index < data.length && sample.length < sampleLimit; index += sampleStride) {
    sample.push(data[index] ?? 0);
  }

  return {
    name: tensor.name,
    description: describeTensor(tensor),
    shape: tensor.shape,
    byteOffset: tensor.byteOffset,
    byteLength: tensor.byteLength,
    elementCount: tensor.data.length,
    min: count > 0 ? min : 0,
    max: count > 0 ? max : 0,
    mean: count > 0 ? sum / count : 0,
    meanAbsolute: count > 0 ? absoluteSum / count : 0,
    sample,
  };
}

function describeTensor(tensor: TensorView): string {
  const { name, shape } = tensor;

  if (name === "transformer.wte.weight") {
    return "Token embedding table that maps vocabulary IDs into hidden-state vectors.";
  }
  if (name === "transformer.wpe.weight") {
    return "Position embedding table that adds sequence-position information to token states.";
  }
  if (name === "transformer.ln_f.weight") {
    return "Scale vector for the final layer normalization before logits are produced.";
  }
  if (name === "transformer.ln_f.bias") {
    return "Bias vector for the final layer normalization before logits are produced.";
  }
  if (name === "lm_head.weight") {
    return "Language-model output projection that maps hidden states back to vocabulary logits.";
  }

  const layerMatch = /^transformer\.h\.(\d+)\.(.+)$/.exec(name);
  if (layerMatch) {
    const layer = Number(layerMatch[1]);
    const suffix = layerMatch[2];
    const layerLabel = `Layer ${layer}`;

    switch (suffix) {
      case "ln_1.weight":
        return `${layerLabel} input layer-normalization scale vector before attention.`;
      case "ln_1.bias":
        return `${layerLabel} input layer-normalization bias vector before attention.`;
      case "attn.attention.k_proj.weight":
        return `${layerLabel} attention key projection matrix.`;
      case "attn.attention.v_proj.weight":
        return `${layerLabel} attention value projection matrix.`;
      case "attn.attention.q_proj.weight":
        return `${layerLabel} attention query projection matrix.`;
      case "attn.attention.out_proj.weight":
        return `${layerLabel} attention output projection matrix back into hidden size.`;
      case "attn.attention.out_proj.bias":
        return `${layerLabel} attention output projection bias vector.`;
      case "ln_2.weight":
        return `${layerLabel} post-attention layer-normalization scale vector before the MLP.`;
      case "ln_2.bias":
        return `${layerLabel} post-attention layer-normalization bias vector before the MLP.`;
      case "mlp.c_fc.weight":
        return `${layerLabel} MLP expansion matrix from hidden size to intermediate size.`;
      case "mlp.c_fc.bias":
        return `${layerLabel} MLP expansion bias vector.`;
      case "mlp.c_proj.weight":
        return `${layerLabel} MLP projection matrix from intermediate size back to hidden size.`;
      case "mlp.c_proj.bias":
        return `${layerLabel} MLP projection bias vector.`;
      default:
        break;
    }
  }

  if (shape.length === 2) {
    return `Matrix tensor with ${shape[0]} rows and ${shape[1]} columns.`;
  }
  if (shape.length === 1) {
    return `Vector tensor with ${shape[0]} elements.`;
  }
  return `Rank-${shape.length} tensor derived from the exported model weights.`;
}

export function allocateRuntimeScratch(
  config: ModelConfig,
  sequenceLength: number,
): RuntimeScratch {
  if (!Number.isInteger(sequenceLength) || sequenceLength < 1) {
    throw new Error(`scratchSequenceLength must be a positive integer, got ${sequenceLength}`);
  }
  if (sequenceLength > config.maximumSequenceLength) {
    throw new Error(
      `scratchSequenceLength ${sequenceLength} exceeds maximumSequenceLength ${config.maximumSequenceLength}`,
    );
  }

  const hiddenElements = sequenceLength * config.hiddenSize;
  const qkvElements = sequenceLength * config.numberOfHeads * config.headDimension;
  const scoreElements = config.numberOfHeads * sequenceLength * sequenceLength;

  return {
    sequenceLength,
    hiddenState: new Float32Array(hiddenElements),
    residual: new Float32Array(hiddenElements),
    normed: new Float32Array(hiddenElements),
    q: new Float32Array(qkvElements),
    k: new Float32Array(qkvElements),
    v: new Float32Array(qkvElements),
    attentionOutput: new Float32Array(hiddenElements),
    attentionScores: new Float32Array(scoreElements),
    mlpIntermediate: new Float32Array(sequenceLength * config.intermediateSize),
    logits: new Float32Array(sequenceLength * config.vocabularySize),
  };
}

export function ensureScratchCapacity(model: LoadedModel, sequenceLength: number): RuntimeScratch {
  if (sequenceLength <= model.scratch.sequenceLength) {
    return model.scratch;
  }

  model.scratch = allocateRuntimeScratch(model.config, sequenceLength);
  return model.scratch;
}

export function validateConfig(config: ModelConfig): void {
  assertObject(config, "config");
  assertEqual(config.architecture, "gpt_neo", "config.architecture");
  assertEqual(config.activation, "gelu_new", "config.activation");
  assertPositiveInteger(config.vocabularySize, "config.vocabularySize");
  assertPositiveInteger(config.hiddenSize, "config.hiddenSize");
  assertPositiveInteger(config.intermediateSize, "config.intermediateSize");
  assertPositiveInteger(config.numberOfLayers, "config.numberOfLayers");
  assertPositiveInteger(config.numberOfHeads, "config.numberOfHeads");
  assertPositiveInteger(config.headDimension, "config.headDimension");
  assertPositiveInteger(config.maximumSequenceLength, "config.maximumSequenceLength");
  assertPositiveNumber(config.layerNormEpsilon, "config.layerNormEpsilon");
  assertPositiveInteger(config.windowSize, "config.windowSize");
  assertNonNegativeInteger(config.bosTokenId, "config.bosTokenId");
  assertNonNegativeInteger(config.eosTokenId, "config.eosTokenId");

  if (config.padTokenId !== null) {
    assertNonNegativeInteger(config.padTokenId, "config.padTokenId");
  }
  if (config.hiddenSize !== config.numberOfHeads * config.headDimension) {
    throw new Error(
      `hiddenSize ${config.hiddenSize} must equal numberOfHeads * headDimension ` +
        `${config.numberOfHeads * config.headDimension}`,
    );
  }
  if (!Array.isArray(config.attentionLayers)) {
    throw new Error("config.attentionLayers must be an array");
  }
  if (config.attentionLayers.length !== config.numberOfLayers) {
    throw new Error(
      `attentionLayers has ${config.attentionLayers.length} entries, expected ${config.numberOfLayers}`,
    );
  }
  for (let i = 0; i < config.attentionLayers.length; i += 1) {
    const attentionKind = config.attentionLayers[i];
    if (attentionKind !== "global" && attentionKind !== "local") {
      throw new Error(`attentionLayers[${i}] must be "global" or "local", got ${attentionKind}`);
    }
  }
}

export function validateWeightsIndex(config: ModelConfig, weightsIndex: WeightsIndex): void {
  assertObject(weightsIndex, "weightsIndex");
  assertEqual(weightsIndex.dtype, "float32", "weightsIndex.dtype");
  assertEqual(weightsIndex.byteOrder, "little-endian", "weightsIndex.byteOrder");
  assertPositiveInteger(weightsIndex.totalByteLength, "weightsIndex.totalByteLength");
  assertPositiveInteger(weightsIndex.tensorCount, "weightsIndex.tensorCount");
  assertObject(weightsIndex.tensors, "weightsIndex.tensors");

  const tensorNames = Object.keys(weightsIndex.tensors);
  if (tensorNames.length !== weightsIndex.tensorCount) {
    throw new Error(
      `tensorCount is ${weightsIndex.tensorCount}, but tensors contains ${tensorNames.length} entries`,
    );
  }

  const expectedShapes = getExpectedTensorShapes(config);
  const expectedNames = new Set(expectedShapes.keys());
  const actualNames = new Set(tensorNames);
  const missing = [...expectedNames].filter((name) => !actualNames.has(name));
  const extra = [...actualNames].filter((name) => !expectedNames.has(name));

  if (missing.length > 0) {
    throw new Error(`weights.json is missing expected tensors: ${missing.join(", ")}`);
  }
  if (extra.length > 0) {
    throw new Error(`weights.json contains unexpected tensors: ${extra.join(", ")}`);
  }

  const spans: Array<{ name: string; start: number; end: number }> = [];
  for (const name of tensorNames) {
    const descriptor = weightsIndex.tensors[name];
    validateTensorDescriptor(name, descriptor);

    const expectedShape = expectedShapes.get(name);
    if (!expectedShape) {
      throw new Error(`No expected shape registered for tensor ${name}`);
    }
    assertShape(name, descriptor.shape, expectedShape);

    const elementCount = product(descriptor.shape);
    const expectedByteLength = elementCount * FLOAT32_BYTES;
    if (descriptor.byteLength !== expectedByteLength) {
      throw new Error(
        `${name} byteLength is ${descriptor.byteLength}, expected ${expectedByteLength}`,
      );
    }

    spans.push({
      name,
      start: descriptor.byteOffset,
      end: descriptor.byteOffset + descriptor.byteLength,
    });
  }

  validateTensorSpans(spans, weightsIndex.totalByteLength);
}

export function validateWeightsBuffer(buffer: ArrayBuffer, weightsIndex: WeightsIndex): void {
  if (buffer.byteLength !== weightsIndex.totalByteLength) {
    throw new Error(
      `weights.bin is ${buffer.byteLength} bytes, weights.json says ${weightsIndex.totalByteLength}`,
    );
  }
}

export function createTensorViews(
  weightsBuffer: ArrayBuffer,
  weightsIndex: WeightsIndex,
): ReadonlyMap<string, TensorView> {
  const tensors = new Map<string, TensorView>();

  for (const [name, descriptor] of Object.entries(weightsIndex.tensors)) {
    tensors.set(name, {
      name,
      shape: Object.freeze([...descriptor.shape]),
      byteOffset: descriptor.byteOffset,
      byteLength: descriptor.byteLength,
      data: new Float32Array(
        weightsBuffer,
        descriptor.byteOffset,
        descriptor.byteLength / FLOAT32_BYTES,
      ),
    });
  }

  return tensors;
}

export function bindModelWeights(
  config: ModelConfig,
  tensors: ReadonlyMap<string, TensorView>,
): BoundModelWeights {
  const layers: TransformerLayerWeights[] = [];

  for (let layer = 0; layer < config.numberOfLayers; layer += 1) {
    const prefix = `transformer.h.${layer}`;
    layers.push({
      index: layer,
      ln1: {
        weight: requireTensor(tensors, `${prefix}.ln_1.weight`),
        bias: requireTensor(tensors, `${prefix}.ln_1.bias`),
      },
      attention: {
        kind: config.attentionLayers[layer],
        kProjWeight: requireTensor(tensors, `${prefix}.attn.attention.k_proj.weight`),
        vProjWeight: requireTensor(tensors, `${prefix}.attn.attention.v_proj.weight`),
        qProjWeight: requireTensor(tensors, `${prefix}.attn.attention.q_proj.weight`),
        outProjWeight: requireTensor(tensors, `${prefix}.attn.attention.out_proj.weight`),
        outProjBias: requireTensor(tensors, `${prefix}.attn.attention.out_proj.bias`),
      },
      ln2: {
        weight: requireTensor(tensors, `${prefix}.ln_2.weight`),
        bias: requireTensor(tensors, `${prefix}.ln_2.bias`),
      },
      mlp: {
        cFcWeight: requireTensor(tensors, `${prefix}.mlp.c_fc.weight`),
        cFcBias: requireTensor(tensors, `${prefix}.mlp.c_fc.bias`),
        cProjWeight: requireTensor(tensors, `${prefix}.mlp.c_proj.weight`),
        cProjBias: requireTensor(tensors, `${prefix}.mlp.c_proj.bias`),
      },
    });
  }

  return {
    tokenEmbedding: requireTensor(tensors, "transformer.wte.weight"),
    positionEmbedding: requireTensor(tensors, "transformer.wpe.weight"),
    layers,
    finalLayerNorm: {
      weight: requireTensor(tensors, "transformer.ln_f.weight"),
      bias: requireTensor(tensors, "transformer.ln_f.bias"),
    },
    lmHead: requireTensor(tensors, "lm_head.weight"),
  };
}

export function getExpectedTensorShapes(config: ModelConfig): Map<string, number[]> {
  const shapes = new Map<string, number[]>();
  const hidden = config.hiddenSize;
  const intermediate = config.intermediateSize;

  shapes.set("transformer.wte.weight", [config.vocabularySize, hidden]);
  shapes.set("transformer.wpe.weight", [config.maximumSequenceLength, hidden]);

  for (let layer = 0; layer < config.numberOfLayers; layer += 1) {
    const prefix = `transformer.h.${layer}`;
    shapes.set(`${prefix}.ln_1.weight`, [hidden]);
    shapes.set(`${prefix}.ln_1.bias`, [hidden]);
    shapes.set(`${prefix}.attn.attention.k_proj.weight`, [hidden, hidden]);
    shapes.set(`${prefix}.attn.attention.v_proj.weight`, [hidden, hidden]);
    shapes.set(`${prefix}.attn.attention.q_proj.weight`, [hidden, hidden]);
    shapes.set(`${prefix}.attn.attention.out_proj.weight`, [hidden, hidden]);
    shapes.set(`${prefix}.attn.attention.out_proj.bias`, [hidden]);
    shapes.set(`${prefix}.ln_2.weight`, [hidden]);
    shapes.set(`${prefix}.ln_2.bias`, [hidden]);
    shapes.set(`${prefix}.mlp.c_fc.weight`, [intermediate, hidden]);
    shapes.set(`${prefix}.mlp.c_fc.bias`, [intermediate]);
    shapes.set(`${prefix}.mlp.c_proj.weight`, [hidden, intermediate]);
    shapes.set(`${prefix}.mlp.c_proj.bias`, [hidden]);
  }

  shapes.set("transformer.ln_f.weight", [hidden]);
  shapes.set("transformer.ln_f.bias", [hidden]);
  shapes.set("lm_head.weight", [config.vocabularySize, hidden]);
  return shapes;
}

async function fetchJson<T>(fetcher: typeof fetch, url: string): Promise<T> {
  const response = await fetcher(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

async function fetchArrayBuffer(
  fetcher: typeof fetch,
  url: string,
  onProgress?: (
    loadedBytes: number,
    totalBytes: number | undefined,
    source: ModelAssetFetchSource,
  ) => void,
): Promise<ArrayBuffer> {
  const response = await fetcher(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }
  const source: ModelAssetFetchSource = isModelAssetCacheHit(response) ? "cache" : "network";
  if (source === "cache") {
    onProgress?.(0, getContentLength(response), "cache");
  }
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    if (source === "network") {
      onProgress?.(buffer.byteLength, getContentLength(response), "network");
    }
    return buffer;
  }

  const totalBytes = getContentLength(response);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    chunks.push(value);
    loadedBytes += value.byteLength;
    if (source === "network") {
      onProgress?.(loadedBytes, totalBytes, "network");
    }
  }

  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (source === "network") {
    onProgress?.(loadedBytes, totalBytes, "network");
  }
  return bytes.buffer;
}

function getContentLength(response: Response): number | undefined {
  const header = response.headers.get("content-length");
  if (!header) {
    return undefined;
  }
  const parsed = Number(header);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

function resolveModelUrl(baseUrl: string, path: string): string {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const absoluteBase = new URL(normalizedBase, globalThis.location?.href).toString();
  return new URL(path, absoluteBase).toString();
}

function validateTensorDescriptor(name: string, descriptor: TensorDescriptor): void {
  assertObject(descriptor, name);
  if (!Array.isArray(descriptor.shape)) {
    throw new Error(`${name}.shape must be an array`);
  }
  for (let index = 0; index < descriptor.shape.length; index += 1) {
    assertPositiveInteger(descriptor.shape[index], `${name}.shape[${index}]`);
  }
  assertNonNegativeInteger(descriptor.byteOffset, `${name}.byteOffset`);
  assertNonNegativeInteger(descriptor.byteLength, `${name}.byteLength`);
  if (descriptor.byteOffset % FLOAT32_BYTES !== 0) {
    throw new Error(`${name}.byteOffset must be ${FLOAT32_BYTES}-byte aligned`);
  }
  if (descriptor.byteLength % FLOAT32_BYTES !== 0) {
    throw new Error(`${name}.byteLength must be a multiple of ${FLOAT32_BYTES}`);
  }
}

function validateTensorSpans(
  spans: Array<{ name: string; start: number; end: number }>,
  totalByteLength: number,
): void {
  spans.sort((left, right) => left.start - right.start);

  let cursor = 0;
  for (const span of spans) {
    if (span.start !== cursor) {
      throw new Error(
        `${span.name} starts at byte ${span.start}, expected contiguous offset ${cursor}`,
      );
    }
    if (span.end > totalByteLength) {
      throw new Error(`${span.name} ends at byte ${span.end}, past total ${totalByteLength}`);
    }
    cursor = span.end;
  }

  if (cursor !== totalByteLength) {
    throw new Error(`Tensor data ends at byte ${cursor}, expected total ${totalByteLength}`);
  }
}

function requireTensor(tensors: ReadonlyMap<string, TensorView>, name: string): TensorView {
  const tensor = tensors.get(name);
  if (!tensor) {
    throw new Error(`Missing tensor ${name}`);
  }
  return tensor;
}

function assertShape(name: string, actual: readonly number[], expected: readonly number[]): void {
  if (actual.length !== expected.length) {
    throw new Error(`${name} rank is ${actual.length}, expected ${expected.length}`);
  }
  for (let i = 0; i < expected.length; i += 1) {
    if (actual[i] !== expected[i]) {
      throw new Error(`${name} shape is [${actual.join(", ")}], expected [${expected.join(", ")}]`);
    }
  }
}

function product(values: readonly number[]): number {
  return values.reduce((result, value) => result * value, 1);
}

function assertObject(value: unknown, name: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
}

function assertEqual<T>(actual: T, expected: T, name: string): void {
  if (actual !== expected) {
    throw new Error(`${name} must be ${String(expected)}, got ${String(actual)}`);
  }
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

function assertNonNegativeInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative integer, got ${String(value)}`);
  }
}

function assertPositiveNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive number, got ${String(value)}`);
  }
}

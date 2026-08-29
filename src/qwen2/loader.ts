import type { TensorView } from "../tensor";
import { isModelAssetCacheHit, type ModelAssetFetchSource } from "../acquisition/modelAssetFetch";
import {
  GGML_TYPE_F32,
  parseGguf,
  readMetadataNumber,
  readMetadataString,
  type GgufFile,
  type GgufMetadataValue,
} from "./gguf";
import {
  createQwenTensorView,
  isFloat32TensorView,
  type QwenTensorView,
} from "./quantizedTensor";

export type { TensorView } from "../tensor";
export type { QwenTensorView } from "./quantizedTensor";

export interface Qwen2Config {
  architecture: "qwen2";
  vocabularySize: number;
  hiddenSize: number;
  intermediateSize: number;
  numberOfLayers: number;
  numberOfHeads: number;
  numberOfKeyValueHeads: number;
  headDimension: number;
  keyValueHiddenSize: number;
  maximumSequenceLength: number;
  rmsNormEpsilon: number;
  ropeTheta: number;
  activation: "silu";
  tiedWordEmbeddings: boolean;
  bosTokenId: number;
  eosTokenId: number;
  padTokenId: number | null;
}

export interface Qwen2NormWeights {
  weight: TensorView;
}

export interface Qwen2AttentionWeights {
  qProjWeight: QwenTensorView;
  qProjBias: TensorView;
  kProjWeight: QwenTensorView;
  kProjBias: TensorView;
  vProjWeight: QwenTensorView;
  vProjBias: TensorView;
  outProjWeight: QwenTensorView;
}

export interface Qwen2MlpWeights {
  gateProjWeight: QwenTensorView;
  upProjWeight: QwenTensorView;
  downProjWeight: QwenTensorView;
}

export interface Qwen2TransformerLayerWeights {
  index: number;
  inputLayerNorm: Qwen2NormWeights;
  attention: Qwen2AttentionWeights;
  postAttentionLayerNorm: Qwen2NormWeights;
  mlp: Qwen2MlpWeights;
}

export interface Qwen2BoundModelWeights {
  tokenEmbedding: QwenTensorView;
  layers: Qwen2TransformerLayerWeights[];
  finalNorm: Qwen2NormWeights;
  lmHead: QwenTensorView;
}

export interface LoadedQwen2Model {
  config: Qwen2Config;
  gguf: GgufFile;
  weightsBuffer: ArrayBuffer;
  tensors: ReadonlyMap<string, QwenTensorView>;
  weights: Qwen2BoundModelWeights;
}

export interface LoadQwen2ModelOptions {
  baseUrl: string;
  ggufPath?: string;
  ggufFallbackUrls?: readonly string[];
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  onProgress?: (progress: Qwen2LoaderProgress) => void;
}

export interface Qwen2LoadedModelSummary {
  architecture: string;
  tensorCount: number;
  totalByteLength: number;
  layers: number;
  hiddenSize: number;
  vocabularySize: number;
  maximumSequenceLength: number;
  config: Qwen2Config;
}

export interface Qwen2LoaderProgress {
  stage:
    | "gguf-download-started"
    | "gguf-download-progress"
    | "gguf-downloaded"
    | "gguf-parsed"
    | "weights-bound"
    | "ready";
  message: string;
  source?: "network" | "cache";
  loadedBytes?: number;
  totalBytes?: number;
}

const DEFAULT_GGUF_PATH = "model.gguf";

export async function loadQwen2Model(options: LoadQwen2ModelOptions): Promise<LoadedQwen2Model> {
  const fetcher = options.fetchImpl ?? fetch;
  const report = options.onProgress ?? (() => undefined);
  const ggufUrls = [
    resolveModelUrl(options.baseUrl, options.ggufPath ?? DEFAULT_GGUF_PATH),
    ...(options.ggufFallbackUrls ?? []),
  ];

  report({
    stage: "gguf-download-started",
    message: "Downloading Qwen2 GGUF file",
  });
  const { buffer: weightsBuffer, url: ggufUrl } = await fetchGgufArrayBuffer(
    fetcher,
    ggufUrls,
    options.signal,
    (loadedBytes, totalBytes, source) => {
      report({
        stage: "gguf-download-progress",
        message: source === "cache" ? "Qwen2 GGUF found in browser cache" : "Downloading Qwen2 GGUF file",
        source,
        loadedBytes,
        totalBytes,
      });
    },
  );
  report({
    stage: "gguf-downloaded",
    message: `Downloaded Qwen2 GGUF file from ${ggufUrl}`,
    loadedBytes: weightsBuffer.byteLength,
    totalBytes: weightsBuffer.byteLength,
  });

  const gguf = parseGguf(weightsBuffer);
  const config = qwen2ConfigFromGguf(gguf);
  validateQwen2Config(config);
  const tensors = createQwenTensorViews(weightsBuffer, gguf);
  validateQwen2TensorSet(config, gguf, tensors);
  report({
    stage: "gguf-parsed",
    message: `Parsed ${gguf.tensors.size} GGUF tensor descriptors`,
    loadedBytes: weightsBuffer.byteLength,
    totalBytes: weightsBuffer.byteLength,
  });

  const weights = bindQwen2ModelWeights(config, tensors);
  report({
    stage: "weights-bound",
    message: "Bound GGUF tensors into typed Qwen2 layers",
  });

  const loadedModel = {
    config,
    gguf,
    weightsBuffer,
    tensors,
    weights,
  };
  report({
    stage: "ready",
    message: "Qwen2 model is ready",
  });
  return loadedModel;
}

export function summarizeLoadedQwen2Model(model: LoadedQwen2Model): Qwen2LoadedModelSummary {
  return {
    architecture: model.config.architecture,
    tensorCount: model.gguf.tensors.size,
    totalByteLength: model.weightsBuffer.byteLength,
    layers: model.config.numberOfLayers,
    hiddenSize: model.config.hiddenSize,
    vocabularySize: model.config.vocabularySize,
    maximumSequenceLength: model.config.maximumSequenceLength,
    config: model.config,
  };
}

export function qwen2ConfigFromGguf(gguf: GgufFile): Qwen2Config {
  const metadata = gguf.metadata;
  const architecture = readRequiredString(metadata, "general.architecture");
  if (architecture !== "qwen2") {
    throw new Error(`Expected qwen2 GGUF architecture, got ${architecture}`);
  }

  const hiddenSize = readRequiredInteger(metadata, "qwen2.embedding_length");
  const numberOfHeads = readRequiredInteger(metadata, "qwen2.attention.head_count");
  const numberOfKeyValueHeads = readRequiredInteger(metadata, "qwen2.attention.head_count_kv");
  const headDimension = hiddenSize / numberOfHeads;
  if (!Number.isInteger(headDimension)) {
    throw new Error(`hiddenSize ${hiddenSize} must divide evenly by numberOfHeads ${numberOfHeads}`);
  }

  return {
    architecture: "qwen2",
    vocabularySize: inferVocabularySize(gguf),
    hiddenSize,
    intermediateSize: readRequiredInteger(metadata, "qwen2.feed_forward_length"),
    numberOfLayers: readRequiredInteger(metadata, "qwen2.block_count"),
    numberOfHeads,
    numberOfKeyValueHeads,
    headDimension,
    keyValueHiddenSize: numberOfKeyValueHeads * headDimension,
    maximumSequenceLength: readRequiredInteger(metadata, "qwen2.context_length"),
    rmsNormEpsilon: readRequiredNumber(metadata, "qwen2.attention.layer_norm_rms_epsilon"),
    ropeTheta: readRequiredNumber(metadata, "qwen2.rope.freq_base"),
    activation: "silu",
    tiedWordEmbeddings: true,
    bosTokenId: readOptionalInteger(metadata, "tokenizer.ggml.bos_token_id") ?? 151643,
    eosTokenId: readOptionalInteger(metadata, "tokenizer.ggml.eos_token_id") ?? 151645,
    padTokenId: readOptionalInteger(metadata, "tokenizer.ggml.padding_token_id"),
  };
}

export function createQwenTensorViews(
  weightsBuffer: ArrayBuffer,
  gguf: GgufFile,
): ReadonlyMap<string, QwenTensorView> {
  const tensors = new Map<string, QwenTensorView>();
  for (const tensor of gguf.tensors.values()) {
    tensors.set(tensor.name, createQwenTensorView(weightsBuffer, tensor));
  }
  return tensors;
}

export function bindQwen2ModelWeights(
  config: Qwen2Config,
  tensors: ReadonlyMap<string, QwenTensorView>,
): Qwen2BoundModelWeights {
  const layers: Qwen2TransformerLayerWeights[] = [];
  for (let layer = 0; layer < config.numberOfLayers; layer += 1) {
    const prefix = `blk.${layer}`;
    layers.push({
      index: layer,
      inputLayerNorm: {
        weight: requireFloat32Tensor(tensors, `${prefix}.attn_norm.weight`),
      },
      attention: {
        qProjWeight: requireTensor(tensors, `${prefix}.attn_q.weight`),
        qProjBias: requireFloat32Tensor(tensors, `${prefix}.attn_q.bias`),
        kProjWeight: requireTensor(tensors, `${prefix}.attn_k.weight`),
        kProjBias: requireFloat32Tensor(tensors, `${prefix}.attn_k.bias`),
        vProjWeight: requireTensor(tensors, `${prefix}.attn_v.weight`),
        vProjBias: requireFloat32Tensor(tensors, `${prefix}.attn_v.bias`),
        outProjWeight: requireTensor(tensors, `${prefix}.attn_output.weight`),
      },
      postAttentionLayerNorm: {
        weight: requireFloat32Tensor(tensors, `${prefix}.ffn_norm.weight`),
      },
      mlp: {
        gateProjWeight: requireTensor(tensors, `${prefix}.ffn_gate.weight`),
        upProjWeight: requireTensor(tensors, `${prefix}.ffn_up.weight`),
        downProjWeight: requireTensor(tensors, `${prefix}.ffn_down.weight`),
      },
    });
  }

  const tokenEmbedding = requireTensor(tensors, "token_embd.weight");
  const lmHead = tensors.get("output.weight") ?? (config.tiedWordEmbeddings ? tokenEmbedding : undefined);
  if (!lmHead) {
    throw new Error("GGUF is missing expected Qwen2 tensor: output.weight");
  }

  return {
    tokenEmbedding,
    layers,
    finalNorm: {
      weight: requireFloat32Tensor(tensors, "output_norm.weight"),
    },
    lmHead,
  };
}

export function validateQwen2Config(config: Qwen2Config): void {
  assertPositiveInteger(config.vocabularySize, "config.vocabularySize");
  assertPositiveInteger(config.hiddenSize, "config.hiddenSize");
  assertPositiveInteger(config.intermediateSize, "config.intermediateSize");
  assertPositiveInteger(config.numberOfLayers, "config.numberOfLayers");
  assertPositiveInteger(config.numberOfHeads, "config.numberOfHeads");
  assertPositiveInteger(config.numberOfKeyValueHeads, "config.numberOfKeyValueHeads");
  assertPositiveInteger(config.headDimension, "config.headDimension");
  assertPositiveInteger(config.maximumSequenceLength, "config.maximumSequenceLength");
  assertPositiveNumber(config.rmsNormEpsilon, "config.rmsNormEpsilon");
  assertPositiveNumber(config.ropeTheta, "config.ropeTheta");
  assertNonNegativeInteger(config.bosTokenId, "config.bosTokenId");
  assertNonNegativeInteger(config.eosTokenId, "config.eosTokenId");
  if (config.padTokenId !== null) {
    assertNonNegativeInteger(config.padTokenId, "config.padTokenId");
  }
  if (config.hiddenSize !== config.numberOfHeads * config.headDimension) {
    throw new Error("Qwen2 hiddenSize must equal numberOfHeads * headDimension");
  }
  if (config.keyValueHiddenSize !== config.numberOfKeyValueHeads * config.headDimension) {
    throw new Error("Qwen2 keyValueHiddenSize must equal numberOfKeyValueHeads * headDimension");
  }
  if (config.numberOfHeads % config.numberOfKeyValueHeads !== 0) {
    throw new Error("Qwen2 numberOfHeads must be divisible by numberOfKeyValueHeads");
  }
  if (config.headDimension % 2 !== 0) {
    throw new Error(`Qwen2 headDimension must be even for RoPE, got ${config.headDimension}`);
  }
  if (config.activation !== "silu") {
    throw new Error(`Qwen2 activation must be silu, got ${config.activation}`);
  }
}

export function validateQwen2TensorSet(
  config: Qwen2Config,
  gguf: GgufFile,
  tensors: ReadonlyMap<string, QwenTensorView>,
): void {
  const expectedShapes = getExpectedQwen2TensorShapes(config);
  for (const [name, expectedShape] of expectedShapes) {
    const tensor = tensors.get(name);
    if (!tensor) {
      throw new Error(`GGUF is missing expected Qwen2 tensor: ${name}`);
    }
    assertShape(name, tensor.shape, expectedShape);
  }
  const outputWeight = tensors.get("output.weight");
  if (outputWeight) {
    assertShape("output.weight", outputWeight.shape, [config.vocabularySize, config.hiddenSize]);
  } else if (!config.tiedWordEmbeddings) {
    throw new Error("GGUF is missing expected Qwen2 tensor: output.weight");
  }
  for (const name of f32TensorNames(config)) {
    if (gguf.tensors.get(name)?.type !== GGML_TYPE_F32) {
      throw new Error(`${name} must be F32`);
    }
  }
}

export function getExpectedQwen2TensorShapes(config: Qwen2Config): Map<string, number[]> {
  const shapes = new Map<string, number[]>();
  const hidden = config.hiddenSize;
  const kvHidden = config.keyValueHiddenSize;
  const intermediate = config.intermediateSize;

  shapes.set("token_embd.weight", [config.vocabularySize, hidden]);
  shapes.set("output_norm.weight", [hidden]);
  for (let layer = 0; layer < config.numberOfLayers; layer += 1) {
    const prefix = `blk.${layer}`;
    shapes.set(`${prefix}.attn_norm.weight`, [hidden]);
    shapes.set(`${prefix}.attn_q.weight`, [hidden, hidden]);
    shapes.set(`${prefix}.attn_q.bias`, [hidden]);
    shapes.set(`${prefix}.attn_k.weight`, [kvHidden, hidden]);
    shapes.set(`${prefix}.attn_k.bias`, [kvHidden]);
    shapes.set(`${prefix}.attn_v.weight`, [kvHidden, hidden]);
    shapes.set(`${prefix}.attn_v.bias`, [kvHidden]);
    shapes.set(`${prefix}.attn_output.weight`, [hidden, hidden]);
    shapes.set(`${prefix}.ffn_norm.weight`, [hidden]);
    shapes.set(`${prefix}.ffn_gate.weight`, [intermediate, hidden]);
    shapes.set(`${prefix}.ffn_up.weight`, [intermediate, hidden]);
    shapes.set(`${prefix}.ffn_down.weight`, [hidden, intermediate]);
  }
  return shapes;
}

function f32TensorNames(config: Qwen2Config): string[] {
  const names = ["output_norm.weight"];
  for (let layer = 0; layer < config.numberOfLayers; layer += 1) {
    const prefix = `blk.${layer}`;
    names.push(
      `${prefix}.attn_norm.weight`,
      `${prefix}.attn_q.bias`,
      `${prefix}.attn_k.bias`,
      `${prefix}.attn_v.bias`,
      `${prefix}.ffn_norm.weight`,
    );
  }
  return names;
}

function inferVocabularySize(gguf: GgufFile): number {
  const output = gguf.tensors.get("output.weight") ?? gguf.tensors.get("token_embd.weight");
  if (!output || output.dimensions.length !== 2) {
    throw new Error("Qwen2 GGUF must include output.weight or token_embd.weight matrix");
  }
  return output.dimensions[1] ?? 0;
}

function requireTensor(
  tensors: ReadonlyMap<string, QwenTensorView>,
  name: string,
): QwenTensorView {
  const tensor = tensors.get(name);
  if (!tensor) {
    throw new Error(`Missing tensor: ${name}`);
  }
  return tensor;
}

function requireFloat32Tensor(
  tensors: ReadonlyMap<string, QwenTensorView>,
  name: string,
): TensorView {
  const tensor = requireTensor(tensors, name);
  if (!isFloat32TensorView(tensor)) {
    throw new Error(`${name} must be F32`);
  }
  return tensor;
}

function readRequiredString(metadata: ReadonlyMap<string, GgufMetadataValue>, key: string): string {
  const value = readMetadataString(metadata, key);
  if (value === null) {
    throw new Error(`GGUF metadata is missing string ${key}`);
  }
  return value;
}

function readRequiredInteger(metadata: ReadonlyMap<string, GgufMetadataValue>, key: string): number {
  const value = readRequiredNumber(metadata, key);
  if (!Number.isInteger(value)) {
    throw new Error(`GGUF metadata ${key} must be an integer, got ${value}`);
  }
  return value;
}

function readOptionalInteger(metadata: ReadonlyMap<string, GgufMetadataValue>, key: string): number | null {
  const value = readMetadataNumber(metadata, key);
  if (value === null) {
    return null;
  }
  if (!Number.isInteger(value)) {
    throw new Error(`GGUF metadata ${key} must be an integer, got ${value}`);
  }
  return value;
}

function readRequiredNumber(metadata: ReadonlyMap<string, GgufMetadataValue>, key: string): number {
  const value = readMetadataNumber(metadata, key);
  if (value === null) {
    throw new Error(`GGUF metadata is missing number ${key}`);
  }
  return value;
}

function assertShape(name: string, actual: readonly number[], expected: readonly number[]): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new Error(`${name} shape is [${actual.join(", ")}], expected [${expected.join(", ")}]`);
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
    throw new Error(`${name} must be a positive finite number, got ${String(value)}`);
  }
}

async function fetchGgufArrayBuffer(
  fetcher: typeof fetch,
  urls: readonly string[],
  signal?: AbortSignal,
  onProgress?: (
    loadedBytes: number,
    totalBytes: number | undefined,
    source: ModelAssetFetchSource,
  ) => void,
): Promise<{ buffer: ArrayBuffer; url: string }> {
  const failures: string[] = [];
  for (const url of urls) {
    try {
      const buffer = await fetchArrayBuffer(fetcher, url, signal, onProgress);
      const magicError = validateGgufMagic(buffer, url);
      if (!magicError) {
        return { buffer, url };
      }
      failures.push(magicError);
    } catch (error: unknown) {
      if (signal?.aborted) {
        throw signal.reason ?? error;
      }
      failures.push(`${url}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  throw new Error(
    "Failed to download a valid Qwen2 GGUF file. Tried:\n" +
      failures.map((failure) => `- ${failure}`).join("\n"),
  );
}

function validateGgufMagic(buffer: ArrayBuffer, url: string): string | null {
  const magic = asciiPrefix(buffer, 4);
  if (magic === "GGUF") {
    return null;
  }

  return (
    `Downloaded ${url}, but it is not a GGUF file (magic ${JSON.stringify(magic)}). ` +
      "The configured Qwen model URL probably returned an HTML page instead of model bytes."
  );
}

function asciiPrefix(buffer: ArrayBuffer, length: number): string {
  const bytes = new Uint8Array(buffer, 0, Math.min(length, buffer.byteLength));
  let value = "";
  for (const byte of bytes) {
    value += String.fromCharCode(byte);
  }
  return value;
}

async function fetchArrayBuffer(
  fetcher: typeof fetch,
  url: string,
  signal?: AbortSignal,
  onProgress?: (
    loadedBytes: number,
    totalBytes: number | undefined,
    source: ModelAssetFetchSource,
  ) => void,
): Promise<ArrayBuffer> {
  const response = await fetcher(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }

  const source = isModelAssetCacheHit(response) ? "cache" : "network";
  const contentLength = response.headers.get("content-length");
  const totalBytes = contentLength ? Number(contentLength) : undefined;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, totalBytes, source);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (value) {
      chunks.push(value);
      loadedBytes += value.byteLength;
      onProgress?.(loadedBytes, totalBytes, source);
    }
  }

  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
}

function resolveModelUrl(baseUrl: string, path: string): string {
  const normalizedBaseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const absoluteBaseUrl = new URL(normalizedBaseUrl, globalThis.location?.href).toString();
  return new URL(path, absoluteBaseUrl).toString();
}

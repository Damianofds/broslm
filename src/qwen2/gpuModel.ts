import type { Qwen2ModelKvCache } from "./attentionCache";
import { resetQwen2ModelKvCache } from "./attentionCache";
import type { BroslmLogger } from "../logger";
import type { LoadedQwen2Model, TensorView } from "./loader";
import { isFloat32TensorView, type QwenTensorView } from "./quantizedTensor";
import {
  planQuantizedMatrixDispatch,
  shouldUseFusedQkvProjection,
  shouldUseSplitAttentionDecode,
} from "./gpuDispatch";
import { qwen2WebGpuPrefillSafetyError } from "./webgpuSafety";
import {
  createStaticStorageBuffer,
  createStorageBuffer,
  clearWebGpuBindGroupCache,
  destroyBuffers,
  encodeComputeShader,
  readMappedGpuBuffer,
  submitWebGpuCommands,
  type WebGpuRuntime,
  webGpuBufferUsage,
} from "../runtime/webgpu";

interface ResidentQwen2GpuContext {
  temps: GPUBuffer[];
  encoder: GPUCommandEncoder;
  dummyBias: GPUBuffer;
  uniformArena?: GPUBuffer;
  uniformOffset: number;
  decodeScratch?: ResidentQwen2GpuDecodeScratch;
}

interface ResidentQwen2GpuDecodeScratch {
  tokenId: GPUBuffer;
  hiddenA: GPUBuffer;
  hiddenB: GPUBuffer;
  normed: GPUBuffer;
  q: GPUBuffer;
  k: GPUBuffer;
  v: GPUBuffer;
  attentionOutput: GPUBuffer;
  gate: GPUBuffer;
  up: GPUBuffer;
  gated: GPUBuffer;
  mlpOutput: GPUBuffer;
  finalHidden: GPUBuffer;
  logits: GPUBuffer;
  attentionPartialStats: GPUBuffer;
  attentionPartialValues: GPUBuffer;
  topKLocalIds: GPUBuffer;
  topKLocalLogits: GPUBuffer;
  topKIds: GPUBuffer;
  topKLogits: GPUBuffer;
  topKReadback: GPUBuffer;
  dummyBias: GPUBuffer;
  uniformArena: GPUBuffer;
  buffers: GPUBuffer[];
}

interface ResidentQwen2GpuLayerCache {
  keys: GPUBuffer;
  values: GPUBuffer;
  length: number;
}

interface ResidentQwen2GpuModelCache {
  runtime: WebGpuRuntime;
  maximumSequenceLength: number;
  hiddenSize: number;
  keyValueHiddenSize: number;
  inputIds: number[];
  layers: ResidentQwen2GpuLayerCache[];
  ropeTable: GPUBuffer;
  kvPrecision: "f16" | "f32";
  decodeScratch: ResidentQwen2GpuDecodeScratch;
}

export interface Qwen2GpuTopKResult {
  tokenIds: Uint32Array;
  logits: Float32Array;
}

const residentQwen2GpuCacheByMetadata = new WeakMap<
  Qwen2ModelKvCache,
  ResidentQwen2GpuModelCache
>();
const qkvBiasDataByQueryBias = new WeakMap<object, Float32Array>();

type Qwen2PrefillProfileStep =
  | "embedding"
  | "attentionNorm"
  | "qkvProjection"
  | "rope"
  | "attention"
  | "attentionOutputProjection"
  | "mlpNorm"
  | "mlpGateProjection"
  | "mlpUpProjection"
  | "mlpActivation"
  | "mlpDownProjection"
  | "residual"
  | "submit"
  | "finalNorm"
  | "logitsProjection"
  | "logitsReadback";

interface Qwen2PrefillProfile {
  startedAt: number;
  promptTokens: number;
  layers: number;
  totals: Partial<Record<Qwen2PrefillProfileStep, number>>;
  fusedQkvLayers: number;
}

export async function qwen2PrefillLogitsResidentGpu(
  model: LoadedQwen2Model,
  inputIds: readonly number[],
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  logger: BroslmLogger,
  topK: number | null,
  basePosition = 0,
): Promise<Qwen2GpuTopKResult | null> {
  const safetyError = qwen2WebGpuPrefillSafetyError(inputIds.length);
  if (safetyError) {
    throw new Error(safetyError);
  }

  let gpuCache: ResidentQwen2GpuModelCache;
  if (basePosition === 0) {
    resetQwen2ModelKvCache(cache);
    gpuCache = resetResidentQwen2GpuCache(model, cache, runtime);
  } else {
    gpuCache = requireResidentQwen2GpuCache(model, cache, runtime);
    if (cache.inputIds.length !== basePosition) {
      throw new Error(
        `Qwen2 prefill chunk starts at ${basePosition}, but cache contains ${cache.inputIds.length} tokens`,
      );
    }
  }

  const sequenceLength = inputIds.length;
  const hiddenSize = model.config.hiddenSize;
  const hiddenElements = sequenceLength * hiddenSize;
  const intermediateElements = sequenceLength * model.config.intermediateSize;
  const keyValueElements = sequenceLength * model.config.keyValueHiddenSize;
  const context = createResidentQwen2GpuContext(runtime);
  const profile = createQwen2PrefillProfile(model, sequenceLength);

  const tokenIdsBuffer = tempBuffer(context, runtime, Uint32Array.from(inputIds));
  const hiddenA = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const hiddenB = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const normed = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const q = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const projectedK = gpuCache.kvPrecision === "f16"
    ? tempBuffer(context, runtime, keyValueElements * Float32Array.BYTES_PER_ELEMENT)
    : null;
  const projectedV = gpuCache.kvPrecision === "f16"
    ? tempBuffer(context, runtime, keyValueElements * Float32Array.BYTES_PER_ELEMENT)
    : null;
  const attentionOutput = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const gate = tempBuffer(context, runtime, intermediateElements * Float32Array.BYTES_PER_ELEMENT);
  const up = tempBuffer(context, runtime, intermediateElements * Float32Array.BYTES_PER_ELEMENT);
  const gated = tempBuffer(context, runtime, intermediateElements * Float32Array.BYTES_PER_ELEMENT);
  const mlpOutput = tempBuffer(context, runtime, hiddenElements * Float32Array.BYTES_PER_ELEMENT);
  const finalHidden = tempBuffer(context, runtime, hiddenSize * Float32Array.BYTES_PER_ELEMENT);
  const logits = tempBuffer(
    context,
    runtime,
    model.config.vocabularySize * Float32Array.BYTES_PER_ELEMENT,
  );

  profileQwen2PrefillStep(profile, "embedding", () => {
    encodeQwen2EmbeddingSequence(context, runtime, {
      tokenIds: tokenIdsBuffer,
      embedding: model.weights.tokenEmbedding,
      output: hiddenA,
      sequenceLength,
      embeddingSize: hiddenSize,
    });
  });

  let layerInput = hiddenA;
  let layerOutput = hiddenB;

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const layerCache = gpuCache.layers[layerIndex];
    if (!layerWeights || !layerCache) {
      throw new Error(`missing Qwen2 layer/GPU cache at index ${layerIndex}`);
    }

    profileQwen2PrefillStep(profile, "attentionNorm", () => {
      encodeRmsNormSequence(context, runtime, {
        input: layerInput,
        weight: tensorBuffer(runtime, layerWeights.inputLayerNorm.weight),
        output: normed,
        sequenceLength,
        featureSize: hiddenSize,
        epsilon: model.config.rmsNormEpsilon,
      });
    });
    profileQwen2PrefillStep(profile, "qkvProjection", () => {
      if (
        encodeQwen2QkvProjectionSequence(context, runtime, {
          input: normed,
          qWeight: layerWeights.attention.qProjWeight,
          qBias: layerWeights.attention.qProjBias,
          kWeight: layerWeights.attention.kProjWeight,
          kBias: layerWeights.attention.kProjBias,
          vWeight: layerWeights.attention.vProjWeight,
          vBias: layerWeights.attention.vProjBias,
          qOutput: q,
          kOutput: projectedK ?? layerCache.keys,
          vOutput: projectedV ?? layerCache.values,
          kOutputBaseOffset: projectedK ? 0 : basePosition * model.config.keyValueHiddenSize,
          vOutputBaseOffset: projectedV ? 0 : basePosition * model.config.keyValueHiddenSize,
          inputSize: hiddenSize,
          qOutputSize: hiddenSize,
          keyValueOutputSize: model.config.keyValueHiddenSize,
          sequenceLength,
        })
      ) {
        profile.fusedQkvLayers += 1;
      }
    });
    profileQwen2PrefillStep(profile, "rope", () => {
      encodeRoPESequence(context, runtime, {
        values: q,
        ropeTable: gpuCache.ropeTable,
        sequenceLength,
        hiddenSize,
        numberOfHeads: model.config.numberOfHeads,
        headDimension: model.config.headDimension,
        basePosition,
        theta: model.config.ropeTheta,
      });
      encodeRoPESequence(context, runtime, {
        values: projectedK ?? layerCache.keys,
        ropeTable: gpuCache.ropeTable,
        valuesBaseOffset: projectedK ? 0 : basePosition * model.config.keyValueHiddenSize,
        sequenceLength,
        hiddenSize: model.config.keyValueHiddenSize,
        numberOfHeads: model.config.numberOfKeyValueHeads,
        headDimension: model.config.headDimension,
        basePosition,
        theta: model.config.ropeTheta,
      });
      if (projectedK && projectedV) {
        encodeF32ToF16(context, runtime, {
          input: projectedK,
          output: layerCache.keys,
          outputOffset: basePosition * model.config.keyValueHiddenSize,
          length: keyValueElements,
        });
        encodeF32ToF16(context, runtime, {
          input: projectedV,
          output: layerCache.values,
          outputOffset: basePosition * model.config.keyValueHiddenSize,
          length: keyValueElements,
        });
      }
    });
    profileQwen2PrefillStep(profile, "attention", () => {
      encodeCausalGqaAttention(context, runtime, {
        q,
        k: layerCache.keys,
        v: layerCache.values,
        output: attentionOutput,
        sequenceLength,
        hiddenSize,
        numberOfHeads: model.config.numberOfHeads,
        numberOfKeyValueHeads: model.config.numberOfKeyValueHeads,
        headDimension: model.config.headDimension,
        keyValueHiddenSize: model.config.keyValueHiddenSize,
        basePosition,
        kvPrecision: gpuCache.kvPrecision,
      });
    });
    profileQwen2PrefillStep(profile, "attentionOutputProjection", () => {
      encodeQwen2MatrixMultiplySequence(context, runtime, {
        input: attentionOutput,
        weight: layerWeights.attention.outProjWeight,
        output: layerOutput,
        inputSize: hiddenSize,
        outputSize: hiddenSize,
        sequenceLength,
      });
    });
    profileQwen2PrefillStep(profile, "residual", () => {
      encodeResidualAdd(context, runtime, {
        left: layerOutput,
        right: layerInput,
        output: attentionOutput,
        length: hiddenElements,
      });
    });

    profileQwen2PrefillStep(profile, "mlpNorm", () => {
      encodeRmsNormSequence(context, runtime, {
        input: attentionOutput,
        weight: tensorBuffer(runtime, layerWeights.postAttentionLayerNorm.weight),
        output: normed,
        sequenceLength,
        featureSize: hiddenSize,
        epsilon: model.config.rmsNormEpsilon,
      });
    });
    profileQwen2PrefillStep(profile, "mlpGateProjection", () => {
      encodeQwen2MatrixMultiplySequence(context, runtime, {
        input: normed,
        weight: layerWeights.mlp.gateProjWeight,
        output: gate,
        inputSize: hiddenSize,
        outputSize: model.config.intermediateSize,
        sequenceLength,
      });
    });
    profileQwen2PrefillStep(profile, "mlpUpProjection", () => {
      encodeQwen2MatrixMultiplySequence(context, runtime, {
        input: normed,
        weight: layerWeights.mlp.upProjWeight,
        output: up,
        inputSize: hiddenSize,
        outputSize: model.config.intermediateSize,
        sequenceLength,
      });
    });
    profileQwen2PrefillStep(profile, "mlpActivation", () => {
      encodeSiluMultiply(context, runtime, {
        input: gate,
        multiplier: up,
        output: gated,
        length: intermediateElements,
      });
    });
    profileQwen2PrefillStep(profile, "mlpDownProjection", () => {
      encodeQwen2MatrixMultiplySequence(context, runtime, {
        input: gated,
        weight: layerWeights.mlp.downProjWeight,
        output: mlpOutput,
        inputSize: model.config.intermediateSize,
        outputSize: hiddenSize,
        sequenceLength,
      });
    });
    profileQwen2PrefillStep(profile, "residual", () => {
      encodeResidualAdd(context, runtime, {
        left: mlpOutput,
        right: attentionOutput,
        output: layerInput,
        length: hiddenElements,
      });
    });
    profileQwen2PrefillStep(profile, "submit", () => {
      submitResidentQwen2GpuContext(runtime, context);
    });
  }

  let sampler: Qwen2TopKBuffers | undefined;
  if (topK !== null) {
    profileQwen2PrefillStep(profile, "finalNorm", () => {
      encodeRmsNormSequence(context, runtime, {
        input: layerInput,
        inputOffset: (sequenceLength - 1) * hiddenSize,
        weight: tensorBuffer(runtime, model.weights.finalNorm.weight),
        output: finalHidden,
        sequenceLength: 1,
        featureSize: hiddenSize,
        epsilon: model.config.rmsNormEpsilon,
      });
    });
    profileQwen2PrefillStep(profile, "logitsProjection", () => {
      encodeQwen2MatrixMultiplySequence(context, runtime, {
        input: finalHidden,
        weight: model.weights.lmHead,
        output: logits,
        inputSize: hiddenSize,
        outputSize: model.config.vocabularySize,
        sequenceLength: 1,
      });
    });
    sampler = encodeTopK(context, runtime, logits, model.config.vocabularySize, topK);
  }

  profileQwen2PrefillStep(profile, "submit", () => {
    submitResidentQwen2GpuContext(runtime, context);
  });
  const topKResult = sampler && topK !== null
    ? await profileQwen2PrefillAsyncStep(profile, "logitsReadback", () =>
        readTopKResult(runtime, sampler, topK),
      )
    : null;

  for (let layerIndex = 0; layerIndex < cache.layers.length; layerIndex += 1) {
    const layerMetadata = cache.layers[layerIndex];
    const gpuLayerCache = gpuCache.layers[layerIndex];
    if (!layerMetadata || !gpuLayerCache) {
      throw new Error(`missing Qwen2 cache at layer ${layerIndex}`);
    }
    layerMetadata.length = basePosition + sequenceLength;
    gpuLayerCache.length = basePosition + sequenceLength;
  }
  cache.inputIds.push(...inputIds);
  gpuCache.inputIds.push(...inputIds);
  destroyBuffers(...context.temps);
  clearWebGpuBindGroupCache(runtime);
  logQwen2PrefillProfile(profile, logger);
  return topKResult;
}

export async function qwen2DecodeTokenLogitsResidentGpu(
  model: LoadedQwen2Model,
  tokenId: number,
  position: number,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  topK: number | null,
): Promise<Qwen2GpuTopKResult | null> {
  const gpuCache = requireResidentQwen2GpuCache(model, cache, runtime);
  if (cache.inputIds.length !== position) {
    throw new Error(`cache input length is ${cache.inputIds.length}, expected position ${position}`);
  }

  const hiddenSize = model.config.hiddenSize;
  const keyValueHiddenSize = model.config.keyValueHiddenSize;
  const intermediateSize = model.config.intermediateSize;
  const scratch = gpuCache.decodeScratch;
  const context = createResidentQwen2GpuContext(runtime, scratch);
  runtime.device.queue.writeBuffer(scratch.tokenId, 0, Uint32Array.of(tokenId));
  const tokenIdsBuffer = scratch.tokenId;
  const hiddenA = scratch.hiddenA;
  const hiddenB = scratch.hiddenB;
  const normed = scratch.normed;
  const q = scratch.q;
  const projectedK = scratch.k;
  const projectedV = scratch.v;
  const attentionOutput = scratch.attentionOutput;
  const gate = scratch.gate;
  const up = scratch.up;
  const gated = scratch.gated;
  const mlpOutput = scratch.mlpOutput;
  const finalHidden = scratch.finalHidden;
  const logits = scratch.logits;

  encodeQwen2EmbeddingSequence(context, runtime, {
    tokenIds: tokenIdsBuffer,
    embedding: model.weights.tokenEmbedding,
    output: hiddenA,
    sequenceLength: 1,
    embeddingSize: hiddenSize,
  });

  let layerInput = hiddenA;
  let layerOutput = hiddenB;

  for (let layerIndex = 0; layerIndex < model.weights.layers.length; layerIndex += 1) {
    const layerWeights = model.weights.layers[layerIndex];
    const gpuLayerCache = gpuCache.layers[layerIndex];
    const layerMetadata = cache.layers[layerIndex];
    if (!layerWeights || !gpuLayerCache || !layerMetadata) {
      throw new Error(`missing Qwen2 layer/cache at index ${layerIndex}`);
    }
    if (gpuLayerCache.length !== position || layerMetadata.length !== position) {
      throw new Error(
        `Qwen2 cache length mismatch at layer ${layerIndex}: GPU ${gpuLayerCache.length}, ` +
          `metadata ${layerMetadata.length}, expected ${position}`,
      );
    }

    encodeRmsNormSequence(context, runtime, {
      input: layerInput,
      weight: tensorBuffer(runtime, layerWeights.inputLayerNorm.weight),
      output: normed,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2QkvProjectionSequence(context, runtime, {
      input: normed,
      qWeight: layerWeights.attention.qProjWeight,
      qBias: layerWeights.attention.qProjBias,
      kWeight: layerWeights.attention.kProjWeight,
      kBias: layerWeights.attention.kProjBias,
      vWeight: layerWeights.attention.vProjWeight,
      vBias: layerWeights.attention.vProjBias,
      qOutput: q,
      kOutput: gpuCache.kvPrecision === "f16" ? projectedK : gpuLayerCache.keys,
      vOutput: gpuCache.kvPrecision === "f16" ? projectedV : gpuLayerCache.values,
      kOutputBaseOffset: gpuCache.kvPrecision === "f16" ? 0 : position * keyValueHiddenSize,
      vOutputBaseOffset: gpuCache.kvPrecision === "f16" ? 0 : position * keyValueHiddenSize,
      inputSize: hiddenSize,
      qOutputSize: hiddenSize,
      keyValueOutputSize: keyValueHiddenSize,
      sequenceLength: 1,
    });
    encodeRoPESequence(context, runtime, {
      values: q,
      ropeTable: gpuCache.ropeTable,
      sequenceLength: 1,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      headDimension: model.config.headDimension,
      basePosition: position,
      theta: model.config.ropeTheta,
    });
    encodeRoPESequence(context, runtime, {
      values: gpuCache.kvPrecision === "f16" ? projectedK : gpuLayerCache.keys,
      ropeTable: gpuCache.ropeTable,
      valuesBaseOffset: gpuCache.kvPrecision === "f16" ? 0 : position * keyValueHiddenSize,
      sequenceLength: 1,
      hiddenSize: keyValueHiddenSize,
      numberOfHeads: model.config.numberOfKeyValueHeads,
      headDimension: model.config.headDimension,
      basePosition: position,
      theta: model.config.ropeTheta,
    });
    if (gpuCache.kvPrecision === "f16") {
      encodeF32ToF16(context, runtime, {
        input: projectedK,
        output: gpuLayerCache.keys,
        outputOffset: position * keyValueHiddenSize,
        length: keyValueHiddenSize,
      });
      encodeF32ToF16(context, runtime, {
        input: projectedV,
        output: gpuLayerCache.values,
        outputOffset: position * keyValueHiddenSize,
        length: keyValueHiddenSize,
      });
    }

    encodeCausalGqaAttentionDecode(context, runtime, {
      q,
      k: gpuLayerCache.keys,
      v: gpuLayerCache.values,
      output: attentionOutput,
      position,
      hiddenSize,
      numberOfHeads: model.config.numberOfHeads,
      numberOfKeyValueHeads: model.config.numberOfKeyValueHeads,
      headDimension: model.config.headDimension,
      keyValueHiddenSize,
      kvPrecision: gpuCache.kvPrecision,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: attentionOutput,
      weight: layerWeights.attention.outProjWeight,
      output: layerOutput,
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });
    encodeResidualAdd(context, runtime, {
      left: layerOutput,
      right: layerInput,
      output: attentionOutput,
      length: hiddenSize,
    });

    encodeRmsNormSequence(context, runtime, {
      input: attentionOutput,
      weight: tensorBuffer(runtime, layerWeights.postAttentionLayerNorm.weight),
      output: normed,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.mlp.gateProjWeight,
      output: gate,
      inputSize: hiddenSize,
      outputSize: intermediateSize,
      sequenceLength: 1,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: normed,
      weight: layerWeights.mlp.upProjWeight,
      output: up,
      inputSize: hiddenSize,
      outputSize: intermediateSize,
      sequenceLength: 1,
    });
    encodeSiluMultiply(context, runtime, {
      input: gate,
      multiplier: up,
      output: gated,
      length: intermediateSize,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: gated,
      weight: layerWeights.mlp.downProjWeight,
      output: mlpOutput,
      inputSize: intermediateSize,
      outputSize: hiddenSize,
      sequenceLength: 1,
    });
    encodeResidualAdd(context, runtime, {
      left: mlpOutput,
      right: attentionOutput,
      output: layerInput,
      length: hiddenSize,
    });
  }

  let sampler: Qwen2TopKBuffers | undefined;
  if (topK !== null) {
    encodeRmsNormSequence(context, runtime, {
      input: layerInput,
      weight: tensorBuffer(runtime, model.weights.finalNorm.weight),
      output: finalHidden,
      sequenceLength: 1,
      featureSize: hiddenSize,
      epsilon: model.config.rmsNormEpsilon,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: finalHidden,
      weight: model.weights.lmHead,
      output: logits,
      inputSize: hiddenSize,
      outputSize: model.config.vocabularySize,
      sequenceLength: 1,
    });
    sampler = encodeTopK(context, runtime, logits, model.config.vocabularySize, topK);
  }

  submitWebGpuCommands(runtime, [context.encoder.finish()]);
  const result = sampler && topK !== null ? await readTopKResult(runtime, sampler, topK) : null;

  for (let layerIndex = 0; layerIndex < cache.layers.length; layerIndex += 1) {
    const layerMetadata = cache.layers[layerIndex];
    const gpuLayerCache = gpuCache.layers[layerIndex];
    if (!layerMetadata || !gpuLayerCache) {
      throw new Error(`missing Qwen2 cache at layer ${layerIndex}`);
    }
    layerMetadata.length = position + 1;
    gpuLayerCache.length = position + 1;
  }
  gpuCache.inputIds.push(tokenId);
  destroyBuffers(...context.temps);
  return result;
}

export function hasQwen2ResidentGpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): boolean {
  const gpuCache = residentQwen2GpuCacheByMetadata.get(cache);
  return Boolean(gpuCache && residentQwen2GpuCacheMatches(model, cache, runtime, gpuCache));
}

function createResidentQwen2GpuContext(
  runtime: WebGpuRuntime,
  decodeScratch?: ResidentQwen2GpuDecodeScratch,
): ResidentQwen2GpuContext {
  const dummyBias = decodeScratch?.dummyBias ?? createStorageBuffer(runtime, Float32Array.of(0));
  return {
    temps: decodeScratch ? [] : [dummyBias],
    encoder: runtime.device.createCommandEncoder(),
    dummyBias,
    uniformArena: decodeScratch?.uniformArena,
    uniformOffset: 0,
    decodeScratch,
  };
}

function submitResidentQwen2GpuContext(
  runtime: WebGpuRuntime,
  context: ResidentQwen2GpuContext,
): void {
  submitWebGpuCommands(runtime, [context.encoder.finish()]);
  context.encoder = runtime.device.createCommandEncoder();
}

function createQwen2PrefillProfile(
  model: LoadedQwen2Model,
  promptTokens: number,
): Qwen2PrefillProfile {
  return {
    startedAt: nowMs(),
    promptTokens,
    layers: model.config.numberOfLayers,
    totals: {},
    fusedQkvLayers: 0,
  };
}

function profileQwen2PrefillStep<T>(
  profile: Qwen2PrefillProfile,
  step: Qwen2PrefillProfileStep,
  fn: () => T,
): T {
  const startedAt = nowMs();
  try {
    return fn();
  } finally {
    addQwen2PrefillTiming(profile, step, nowMs() - startedAt);
  }
}

async function profileQwen2PrefillAsyncStep<T>(
  profile: Qwen2PrefillProfile,
  step: Qwen2PrefillProfileStep,
  fn: () => Promise<T>,
): Promise<T> {
  const startedAt = nowMs();
  try {
    return await fn();
  } finally {
    addQwen2PrefillTiming(profile, step, nowMs() - startedAt);
  }
}

function addQwen2PrefillTiming(
  profile: Qwen2PrefillProfile,
  step: Qwen2PrefillProfileStep,
  elapsedMs: number,
): void {
  profile.totals[step] = (profile.totals[step] ?? 0) + elapsedMs;
}

function logQwen2PrefillProfile(profile: Qwen2PrefillProfile, logger: BroslmLogger): void {
  logger.debug("webgpu-prefill-profile", {
    promptTokens: profile.promptTokens,
    layers: profile.layers,
    fusedQkvLayers: profile.fusedQkvLayers,
    totalWallMs: roundTiming(nowMs() - profile.startedAt),
    encodeAndSyncMs: Object.fromEntries(
      Object.entries(profile.totals).map(([step, value]) => [step, roundTiming(value ?? 0)]),
    ),
  });
}

function roundTiming(value: number): number {
  return Math.round(value * 10) / 10;
}

function nowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function tempBuffer(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  dataOrByteLength: Float32Array | Uint32Array | Uint8Array | ArrayBuffer | number,
  usage?: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = createStorageBuffer(runtime, dataOrByteLength, usage);
  context.temps.push(buffer);
  return buffer;
}

function getOrCreateResidentQwen2GpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuModelCache {
  const existing = residentQwen2GpuCacheByMetadata.get(cache);
  if (existing && residentQwen2GpuCacheShapeMatches(model, cache, runtime, existing)) {
    return existing;
  }
  if (existing) {
    destroyResidentQwen2GpuCache(existing);
  }

  const kvPrecision = runtime.shaderF16 === true ? "f16" : "f32";
  const layerByteLength =
    cache.maximumSequenceLength * model.config.keyValueHiddenSize *
    (kvPrecision === "f16" ? Uint16Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT);
  const gpuCache: ResidentQwen2GpuModelCache = {
    runtime,
    maximumSequenceLength: cache.maximumSequenceLength,
    hiddenSize: model.config.hiddenSize,
    keyValueHiddenSize: model.config.keyValueHiddenSize,
    inputIds: [],
    layers: Array.from({ length: model.config.numberOfLayers }, () => ({
      keys: createStorageBuffer(runtime, layerByteLength),
      values: createStorageBuffer(runtime, layerByteLength),
      length: 0,
    })),
    ropeTable: createQwen2RoPETable(model, cache, runtime),
    kvPrecision,
    decodeScratch: createResidentQwen2GpuDecodeScratch(model, cache, runtime),
  };
  residentQwen2GpuCacheByMetadata.set(cache, gpuCache);
  return gpuCache;
}

function resetResidentQwen2GpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuModelCache {
  const gpuCache = getOrCreateResidentQwen2GpuCache(model, cache, runtime);
  gpuCache.inputIds.length = 0;
  for (const layer of gpuCache.layers) {
    layer.length = 0;
  }
  return gpuCache;
}

function requireResidentQwen2GpuCache(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuModelCache {
  const gpuCache = residentQwen2GpuCacheByMetadata.get(cache);
  if (!gpuCache || !residentQwen2GpuCacheMatches(model, cache, runtime, gpuCache)) {
    throw new Error("Qwen2 WebGPU decode requires a resident GPU cache initialized by prefill.");
  }
  return gpuCache;
}

function residentQwen2GpuCacheMatches(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  gpuCache: ResidentQwen2GpuModelCache,
): boolean {
  if (!residentQwen2GpuCacheShapeMatches(model, cache, runtime, gpuCache)) {
    return false;
  }
  if (gpuCache.inputIds.length !== cache.inputIds.length) {
    return false;
  }
  for (let index = 0; index < cache.inputIds.length; index += 1) {
    if (gpuCache.inputIds[index] !== cache.inputIds[index]) {
      return false;
    }
  }
  for (let layerIndex = 0; layerIndex < cache.layers.length; layerIndex += 1) {
    const layerMetadata = cache.layers[layerIndex];
    const gpuLayer = gpuCache.layers[layerIndex];
    if (!layerMetadata || !gpuLayer || layerMetadata.length !== gpuLayer.length) {
      return false;
    }
  }
  return true;
}

function residentQwen2GpuCacheShapeMatches(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
  gpuCache: ResidentQwen2GpuModelCache,
): boolean {
  return (
    gpuCache.runtime === runtime &&
    gpuCache.maximumSequenceLength === cache.maximumSequenceLength &&
    gpuCache.hiddenSize === model.config.hiddenSize &&
    gpuCache.keyValueHiddenSize === model.config.keyValueHiddenSize &&
    gpuCache.layers.length === model.config.numberOfLayers
  );
}

function destroyResidentQwen2GpuCache(gpuCache: ResidentQwen2GpuModelCache): void {
  for (const layer of gpuCache.layers) {
    destroyBuffers(layer.keys, layer.values);
  }
  destroyBuffers(gpuCache.ropeTable);
  destroyBuffers(...gpuCache.decodeScratch.buffers);
}

function createQwen2RoPETable(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): GPUBuffer {
  const halfDimension = model.config.headDimension / 2;
  const values = new Float32Array(cache.maximumSequenceLength * halfDimension * 2);
  for (let position = 0; position < cache.maximumSequenceLength; position += 1) {
    for (let pair = 0; pair < halfDimension; pair += 1) {
      const angle = position / Math.pow(model.config.ropeTheta, (2 * pair) / model.config.headDimension);
      const offset = (position * halfDimension + pair) * 2;
      values[offset] = Math.cos(angle);
      values[offset + 1] = Math.sin(angle);
    }
  }
  return createStorageBuffer(runtime, values);
}

function createResidentQwen2GpuDecodeScratch(
  model: LoadedQwen2Model,
  cache: Qwen2ModelKvCache,
  runtime: WebGpuRuntime,
): ResidentQwen2GpuDecodeScratch {
  const buffers: GPUBuffer[] = [];
  const allocate = (
    byteLength: number,
    usage?: GPUBufferUsageFlags,
  ): GPUBuffer => {
    const buffer = createStorageBuffer(runtime, byteLength, usage);
    buffers.push(buffer);
    return buffer;
  };
  const floatBytes = Float32Array.BYTES_PER_ELEMENT;
  const hiddenBytes = model.config.hiddenSize * floatBytes;
  const keyValueBytes = model.config.keyValueHiddenSize * floatBytes;
  const intermediateBytes = model.config.intermediateSize * floatBytes;
  const maximumTiles = Math.ceil(cache.maximumSequenceLength / 256);
  const maximumTopK = 64;
  const localTopKElements = 256 * maximumTopK;

  const scratch: Omit<ResidentQwen2GpuDecodeScratch, "buffers"> = {
    tokenId: allocate(Uint32Array.BYTES_PER_ELEMENT),
    hiddenA: allocate(hiddenBytes),
    hiddenB: allocate(hiddenBytes),
    normed: allocate(hiddenBytes),
    q: allocate(hiddenBytes),
    k: allocate(keyValueBytes),
    v: allocate(keyValueBytes),
    attentionOutput: allocate(hiddenBytes),
    gate: allocate(intermediateBytes),
    up: allocate(intermediateBytes),
    gated: allocate(intermediateBytes),
    mlpOutput: allocate(hiddenBytes),
    finalHidden: allocate(hiddenBytes),
    logits: allocate(model.config.vocabularySize * floatBytes),
    attentionPartialStats: allocate(model.config.numberOfHeads * maximumTiles * 2 * floatBytes),
    attentionPartialValues: allocate(
      model.config.numberOfHeads * maximumTiles * model.config.headDimension * floatBytes,
    ),
    topKLocalIds: allocate(localTopKElements * Uint32Array.BYTES_PER_ELEMENT),
    topKLocalLogits: allocate(localTopKElements * floatBytes),
    topKIds: allocate(maximumTopK * Uint32Array.BYTES_PER_ELEMENT),
    topKLogits: allocate(maximumTopK * floatBytes),
    topKReadback: allocate(
      maximumTopK * (Uint32Array.BYTES_PER_ELEMENT + floatBytes),
      webGpuBufferUsage.copyDst | webGpuBufferUsage.mapRead,
    ),
    dummyBias: allocate(floatBytes),
    uniformArena: allocate(
      256 * 512,
      webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
    ),
  };
  return { ...scratch, buffers };
}

function tensorBuffer(runtime: WebGpuRuntime, tensor: TensorView): GPUBuffer {
  return createStaticStorageBuffer(runtime, tensor.data);
}

function qwenTensorBuffer(runtime: WebGpuRuntime, tensor: QwenTensorView): GPUBuffer {
  return createStaticStorageBuffer(runtime, tensor.data);
}

function biasBuffer(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  bias: TensorView | Float32Array | undefined,
): { buffer: GPUBuffer; hasBias: boolean } {
  if (!bias) {
    return { buffer: context.dummyBias, hasBias: false };
  }
  if (bias instanceof Float32Array) {
    return { buffer: tempBuffer(context, runtime, bias), hasBias: true };
  }
  return { buffer: tensorBuffer(runtime, bias), hasBias: true };
}

function paramsBuffer(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  data: Float32Array | Uint32Array,
): GPUBufferBinding {
  if (!context.uniformArena) {
    return {
      buffer: tempBuffer(
        context,
        runtime,
        data,
        webGpuBufferUsage.uniform | webGpuBufferUsage.copyDst,
      ),
    };
  }
  const offset = Math.ceil(context.uniformOffset / 256) * 256;
  if (offset + data.byteLength > 256 * 512) {
    throw new Error("Qwen2 decode uniform arena is exhausted.");
  }
  runtime.device.queue.writeBuffer(context.uniformArena, offset, data);
  context.uniformOffset = offset + 256;
  return { buffer: context.uniformArena, offset, size: data.byteLength };
}

interface Qwen2TopKBuffers {
  tokenIds: GPUBuffer;
  logits: GPUBuffer;
  readback: GPUBuffer;
  readbackByteLength: number;
  logitsReadbackOffset?: number;
}

function encodeTopK(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  logits: GPUBuffer,
  vocabularySize: number,
  requestedTopK: number,
): Qwen2TopKBuffers {
  const topK = Math.max(1, Math.min(64, Math.round(requestedTopK)));
  const laneCount = 256;
  const tokenIds = context.decodeScratch?.topKIds ??
    tempBuffer(context, runtime, topK * Uint32Array.BYTES_PER_ELEMENT);
  const selectedLogits = context.decodeScratch?.topKLogits ??
    tempBuffer(context, runtime, topK * Float32Array.BYTES_PER_ELEMENT);
  const readback = context.decodeScratch?.topKReadback ?? tempBuffer(
    context,
    runtime,
    64 * (Uint32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT),
    webGpuBufferUsage.copyDst | webGpuBufferUsage.mapRead,
  );
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([vocabularySize, topK, laneCount * topK, 0]),
  );
  if (topK === 1) {
    encodeComputeShader(
      runtime,
      context.encoder,
      qwen2ArgmaxShader,
      [
        { binding: 0, resource: { buffer: logits } },
        { binding: 1, resource: params },
        { binding: 2, resource: { buffer: tokenIds } },
        { binding: 3, resource: { buffer: selectedLogits } },
      ],
      [1],
    );
    context.encoder.copyBufferToBuffer(
      tokenIds,
      0,
      readback,
      0,
      Uint32Array.BYTES_PER_ELEMENT,
    );
    return {
      tokenIds,
      logits: selectedLogits,
      readback,
      readbackByteLength: Uint32Array.BYTES_PER_ELEMENT,
    };
  }
  const localIds = context.decodeScratch?.topKLocalIds ??
    tempBuffer(context, runtime, laneCount * topK * Uint32Array.BYTES_PER_ELEMENT);
  const localLogits = context.decodeScratch?.topKLocalLogits ??
    tempBuffer(context, runtime, laneCount * topK * Float32Array.BYTES_PER_ELEMENT);
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2TopKLocalShader,
    [
      { binding: 0, resource: { buffer: logits } },
      { binding: 1, resource: params },
      { binding: 2, resource: { buffer: localIds } },
      { binding: 3, resource: { buffer: localLogits } },
    ],
    [1],
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2TopKMergeShader,
    [
      { binding: 0, resource: { buffer: localIds } },
      { binding: 1, resource: { buffer: localLogits } },
      { binding: 2, resource: params },
      { binding: 3, resource: { buffer: tokenIds } },
      { binding: 4, resource: { buffer: selectedLogits } },
    ],
    [1],
  );
  const idsByteLength = topK * Uint32Array.BYTES_PER_ELEMENT;
  const logitsByteLength = topK * Float32Array.BYTES_PER_ELEMENT;
  context.encoder.copyBufferToBuffer(tokenIds, 0, readback, 0, idsByteLength);
  context.encoder.copyBufferToBuffer(
    selectedLogits,
    0,
    readback,
    idsByteLength,
    logitsByteLength,
  );
  return {
    tokenIds,
    logits: selectedLogits,
    readback,
    readbackByteLength: idsByteLength + logitsByteLength,
    logitsReadbackOffset: idsByteLength,
  };
}

async function readTopKResult(
  runtime: WebGpuRuntime,
  buffers: Qwen2TopKBuffers,
  requestedTopK: number,
): Promise<Qwen2GpuTopKResult> {
  const topK = Math.max(1, Math.min(64, Math.round(requestedTopK)));
  if (buffers.readback) {
    const idsByteLength = topK * Uint32Array.BYTES_PER_ELEMENT;
    const data = await readMappedGpuBuffer(runtime, buffers.readback, buffers.readbackByteLength);
    if (topK === 1) {
      return {
        tokenIds: new Uint32Array(data),
        logits: Float32Array.of(0),
      };
    }
    const logitsOffset = buffers.logitsReadbackOffset ?? idsByteLength;
    const logitsByteLength = topK * Float32Array.BYTES_PER_ELEMENT;
    return {
      tokenIds: new Uint32Array(data.slice(0, idsByteLength)),
      logits: new Float32Array(data.slice(logitsOffset, logitsOffset + logitsByteLength)),
    };
  }
  throw new Error("Qwen2 top-k readback buffer is unavailable");
}

function encodeQwen2EmbeddingSequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    tokenIds: GPUBuffer;
    embedding: QwenTensorView;
    output: GPUBuffer;
    sequenceLength: number;
    embeddingSize: number;
  },
): void {
  const [entryCount, embeddingSize] = requireMatrixShape(options.embedding, "embedding");
  if (embeddingSize !== options.embeddingSize) {
    throw new Error(`embedding size is ${embeddingSize}, expected ${options.embeddingSize}`);
  }

  if (isFloat32TensorView(options.embedding)) {
    const params = paramsBuffer(
      context,
      runtime,
      new Uint32Array([options.sequenceLength, embeddingSize, entryCount, 0]),
    );
    encodeComputeShader(
      runtime,
      context.encoder,
      qwen2F32EmbeddingSequenceShader,
      [
        { binding: 0, resource: { buffer: options.tokenIds } },
        { binding: 1, resource: { buffer: qwenTensorBuffer(runtime, options.embedding) } },
        { binding: 2, resource: params },
        { binding: 3, resource: { buffer: options.output } },
      ],
      [Math.ceil((options.sequenceLength * embeddingSize) / 128)],
    );
    return;
  }

  const rowByteLength = quantizedRowByteLength(options.embedding, embeddingSize);
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.sequenceLength,
      embeddingSize,
      rowByteLength,
      quantizedTypeCode(options.embedding),
      entryCount,
      0,
      0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2QuantizedEmbeddingSequenceShader,
    [
      { binding: 0, resource: { buffer: options.tokenIds } },
      { binding: 1, resource: { buffer: qwenTensorBuffer(runtime, options.embedding) } },
      { binding: 2, resource: params },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil((options.sequenceLength * embeddingSize) / 128)],
  );
}

function encodeQwen2MatrixMultiplySequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputBaseOffset?: number;
    weight: QwenTensorView;
    bias?: TensorView | Float32Array;
    output: GPUBuffer;
    outputBaseOffset?: number;
    inputSize: number;
    outputSize: number;
    sequenceLength: number;
  },
): void {
  const [weightOutputSize, weightInputSize] = requireMatrixShape(options.weight, "weight");
  if (weightInputSize !== options.inputSize || weightOutputSize !== options.outputSize) {
    throw new Error(
      `weight shape is [${weightOutputSize}, ${weightInputSize}], expected ` +
        `[${options.outputSize}, ${options.inputSize}]`,
    );
  }
  const resolvedBias = biasBuffer(context, runtime, options.bias);

  if (isFloat32TensorView(options.weight)) {
    const params = paramsBuffer(
      context,
      runtime,
      new Uint32Array([
        options.inputSize,
        options.outputSize,
        options.sequenceLength,
        options.inputBaseOffset ?? 0,
        resolvedBias.hasBias ? 1 : 0,
        0,
        0,
        options.outputBaseOffset ?? 0,
      ]),
    );
    encodeComputeShader(
      runtime,
      context.encoder,
      qwen2F32MatrixMultiplySequenceShader,
      [
        { binding: 0, resource: { buffer: qwenTensorBuffer(runtime, options.weight) } },
        { binding: 1, resource: { buffer: options.input } },
        { binding: 2, resource: { buffer: resolvedBias.buffer } },
        { binding: 3, resource: params },
        { binding: 4, resource: { buffer: options.output } },
      ],
      [Math.ceil((options.sequenceLength * options.outputSize) / 64)],
    );
    return;
  }

  const rowByteLength = quantizedRowByteLength(options.weight, options.inputSize);
  const dispatch = planQuantizedMatrixDispatch(
    options.outputSize,
    options.sequenceLength,
    runtime.device.limits.maxComputeWorkgroupsPerDimension,
  );
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.inputSize,
      options.outputSize,
      options.sequenceLength,
      options.inputBaseOffset ?? 0,
      rowByteLength,
      quantizedTypeCode(options.weight),
      resolvedBias.hasBias ? 1 : 0,
      options.outputBaseOffset ?? 0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2QuantizedMatrixMultiplySequenceShader,
    [
      { binding: 0, resource: { buffer: qwenTensorBuffer(runtime, options.weight) } },
      { binding: 1, resource: { buffer: options.input } },
      { binding: 2, resource: { buffer: resolvedBias.buffer } },
      { binding: 3, resource: params },
      { binding: 4, resource: { buffer: options.output } },
    ],
    dispatch.workgroups,
  );
}

function encodeQwen2QkvProjectionSequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputBaseOffset?: number;
    qWeight: QwenTensorView;
    qBias?: TensorView | Float32Array;
    kWeight: QwenTensorView;
    kBias?: TensorView | Float32Array;
    vWeight: QwenTensorView;
    vBias?: TensorView | Float32Array;
    qOutput: GPUBuffer;
    kOutput: GPUBuffer;
    vOutput: GPUBuffer;
    qOutputBaseOffset?: number;
    kOutputBaseOffset?: number;
    vOutputBaseOffset?: number;
    inputSize: number;
    qOutputSize: number;
    keyValueOutputSize: number;
    sequenceLength: number;
  },
): boolean {
  validateQkvProjectionShapes(options);
  // A single QKV dispatch avoids two pipeline launches whenever all three
  // outputs start at zero. Offset writes into an f32 KV cache stay separate.
  if (!shouldUseFusedQkvProjection({
    sequenceLength: options.sequenceLength,
    qOutputBaseOffset: options.qOutputBaseOffset,
    kOutputBaseOffset: options.kOutputBaseOffset,
    vOutputBaseOffset: options.vOutputBaseOffset,
    weightsCompatible: canFuseQwen2QkvProjection(
      options.qWeight,
      options.kWeight,
      options.vWeight,
    ),
  })) {
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: options.input,
      inputBaseOffset: options.inputBaseOffset,
      weight: options.qWeight,
      bias: options.qBias,
      output: options.qOutput,
      outputBaseOffset: options.qOutputBaseOffset,
      inputSize: options.inputSize,
      outputSize: options.qOutputSize,
      sequenceLength: options.sequenceLength,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: options.input,
      inputBaseOffset: options.inputBaseOffset,
      weight: options.kWeight,
      bias: options.kBias,
      output: options.kOutput,
      outputBaseOffset: options.kOutputBaseOffset,
      inputSize: options.inputSize,
      outputSize: options.keyValueOutputSize,
      sequenceLength: options.sequenceLength,
    });
    encodeQwen2MatrixMultiplySequence(context, runtime, {
      input: options.input,
      inputBaseOffset: options.inputBaseOffset,
      weight: options.vWeight,
      bias: options.vBias,
      output: options.vOutput,
      outputBaseOffset: options.vOutputBaseOffset,
      inputSize: options.inputSize,
      outputSize: options.keyValueOutputSize,
      sequenceLength: options.sequenceLength,
    });
    return false;
  }

  const qkvBias = qkvBiasBuffer(context, runtime, {
    qBias: options.qBias,
    kBias: options.kBias,
    vBias: options.vBias,
    qOutputSize: options.qOutputSize,
    keyValueOutputSize: options.keyValueOutputSize,
  });

  if (isFloat32TensorView(options.qWeight)) {
    const params = paramsBuffer(
      context,
      runtime,
      new Uint32Array([
        options.inputSize,
        options.qOutputSize,
        options.keyValueOutputSize,
        options.sequenceLength,
        options.inputBaseOffset ?? 0,
        0,
        0,
        0,
      ]),
    );
    encodeComputeShader(
      runtime,
      context.encoder,
      qwen2F32QkvProjectionSequenceShader,
      [
        { binding: 0, resource: { buffer: qwenTensorBuffer(runtime, options.qWeight) } },
        { binding: 1, resource: { buffer: qwenTensorBuffer(runtime, options.kWeight) } },
        { binding: 2, resource: { buffer: qwenTensorBuffer(runtime, options.vWeight) } },
        { binding: 3, resource: { buffer: options.input } },
        { binding: 4, resource: { buffer: qkvBias } },
        { binding: 5, resource: params },
        { binding: 6, resource: { buffer: options.qOutput } },
        { binding: 7, resource: { buffer: options.kOutput } },
        { binding: 8, resource: { buffer: options.vOutput } },
      ],
      [
        Math.ceil(
          (options.sequenceLength *
            (options.qOutputSize + 2 * options.keyValueOutputSize)) /
            64,
        ),
      ],
    );
    return true;
  }

  if (
    isFloat32TensorView(options.kWeight) ||
    isFloat32TensorView(options.vWeight) ||
    options.qWeight.type !== options.kWeight.type ||
    options.qWeight.type !== options.vWeight.type
  ) {
    throw new Error("invalid Qwen2 QKV fusion state");
  }

  const rowByteLength = quantizedRowByteLength(options.qWeight, options.inputSize);
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.inputSize,
      options.qOutputSize,
      options.keyValueOutputSize,
      options.sequenceLength,
      options.inputBaseOffset ?? 0,
      rowByteLength,
      quantizedTypeCode(options.qWeight),
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2QuantizedQkvProjectionSequenceShader,
    [
      { binding: 0, resource: { buffer: qwenTensorBuffer(runtime, options.qWeight) } },
      { binding: 1, resource: { buffer: qwenTensorBuffer(runtime, options.kWeight) } },
      { binding: 2, resource: { buffer: qwenTensorBuffer(runtime, options.vWeight) } },
      { binding: 3, resource: { buffer: options.input } },
      { binding: 4, resource: { buffer: qkvBias } },
      { binding: 5, resource: params },
      { binding: 6, resource: { buffer: options.qOutput } },
      { binding: 7, resource: { buffer: options.kOutput } },
      { binding: 8, resource: { buffer: options.vOutput } },
    ],
    [
      Math.ceil(
        (options.sequenceLength *
          (options.qOutputSize + 2 * options.keyValueOutputSize)) /
          64,
      ),
    ],
  );
  return true;
}

function encodeRmsNormSequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    inputOffset?: number;
    weight: GPUBuffer;
    output: GPUBuffer;
    sequenceLength: number;
    featureSize: number;
    epsilon: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Float32Array([
      options.epsilon,
      options.inputOffset ?? 0,
      options.featureSize,
      options.sequenceLength,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2RmsNormSequenceShader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: { buffer: options.weight } },
      { binding: 2, resource: params },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [options.sequenceLength],
  );
}

function encodeRoPESequence(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    values: GPUBuffer;
    ropeTable: GPUBuffer;
    valuesBaseOffset?: number;
    sequenceLength: number;
    hiddenSize: number;
    numberOfHeads: number;
    headDimension: number;
    basePosition: number;
    theta: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Float32Array([
      options.theta,
      options.sequenceLength,
      options.hiddenSize,
      options.numberOfHeads,
      options.headDimension,
      options.basePosition,
      options.valuesBaseOffset ?? 0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2RoPESequenceShader,
    [
      { binding: 0, resource: { buffer: options.values } },
      { binding: 1, resource: params },
      { binding: 2, resource: { buffer: options.ropeTable } },
    ],
    [Math.ceil((options.sequenceLength * options.numberOfHeads * (options.headDimension / 2)) / 128)],
  );
}

function encodeCausalGqaAttention(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    q: GPUBuffer;
    k: GPUBuffer;
    v: GPUBuffer;
    output: GPUBuffer;
    sequenceLength: number;
    hiddenSize: number;
    numberOfHeads: number;
    numberOfKeyValueHeads: number;
    headDimension: number;
    keyValueHiddenSize: number;
    basePosition?: number;
    kvPrecision: "f16" | "f32";
  },
): void {
  if (options.headDimension !== 64) {
    throw new Error(`prefill attention requires Qwen2.5 headDimension 64, got ${options.headDimension}`);
  }
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.sequenceLength,
      options.hiddenSize,
      options.numberOfHeads,
      options.numberOfKeyValueHeads,
      options.headDimension,
      options.keyValueHiddenSize,
      options.basePosition ?? 0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    options.kvPrecision === "f16"
      ? qwen2CausalGqaAttentionF16Shader
      : qwen2CausalGqaAttentionShader,
    [
      { binding: 0, resource: { buffer: options.q } },
      { binding: 1, resource: { buffer: options.k } },
      { binding: 2, resource: { buffer: options.v } },
      { binding: 3, resource: params },
      { binding: 4, resource: { buffer: options.output } },
    ],
    [options.sequenceLength, options.numberOfHeads],
  );
}

function encodeCausalGqaAttentionDecode(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    q: GPUBuffer;
    k: GPUBuffer;
    v: GPUBuffer;
    output: GPUBuffer;
    position: number;
    hiddenSize: number;
    numberOfHeads: number;
    numberOfKeyValueHeads: number;
    headDimension: number;
    keyValueHiddenSize: number;
    kvPrecision: "f16" | "f32";
  },
): void {
  if (options.headDimension !== 64) {
    throw new Error(`decode attention requires Qwen2.5 headDimension 64, got ${options.headDimension}`);
  }
  const tileSize = 256;
  if (!shouldUseSplitAttentionDecode(options.position, tileSize)) {
    const params = paramsBuffer(
      context,
      runtime,
      new Uint32Array([
        1,
        options.hiddenSize,
        options.numberOfHeads,
        options.numberOfKeyValueHeads,
        options.headDimension,
        options.keyValueHiddenSize,
        options.position,
        0,
      ]),
    );
    encodeComputeShader(
      runtime,
      context.encoder,
      options.kvPrecision === "f16"
        ? qwen2CausalGqaAttentionF16Shader
        : qwen2CausalGqaAttentionShader,
      [
        { binding: 0, resource: { buffer: options.q } },
        { binding: 1, resource: { buffer: options.k } },
        { binding: 2, resource: { buffer: options.v } },
        { binding: 3, resource: params },
        { binding: 4, resource: { buffer: options.output } },
      ],
      [1, options.numberOfHeads],
    );
    return;
  }
  const tileCount = Math.ceil((options.position + 1) / tileSize);
  const partialStats = context.decodeScratch?.attentionPartialStats ?? tempBuffer(
    context,
    runtime,
    options.numberOfHeads * tileCount * 2 * Float32Array.BYTES_PER_ELEMENT,
  );
  const partialValues = context.decodeScratch?.attentionPartialValues ?? tempBuffer(
    context,
    runtime,
    options.numberOfHeads * tileCount * options.headDimension * Float32Array.BYTES_PER_ELEMENT,
  );
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.position,
      options.numberOfHeads,
      options.numberOfKeyValueHeads,
      options.headDimension,
      options.keyValueHiddenSize,
      tileSize,
      tileCount,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    options.kvPrecision === "f16"
      ? qwen2CausalGqaAttentionDecodeF16Shader
      : qwen2CausalGqaAttentionDecodeShader,
    [
      { binding: 0, resource: { buffer: options.q } },
      { binding: 1, resource: { buffer: options.k } },
      { binding: 2, resource: { buffer: options.v } },
      { binding: 3, resource: params },
      { binding: 4, resource: { buffer: partialStats } },
      { binding: 5, resource: { buffer: partialValues } },
    ],
    [tileCount, options.numberOfHeads],
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2CausalGqaAttentionDecodeMergeShader,
    [
      { binding: 0, resource: { buffer: partialStats } },
      { binding: 1, resource: { buffer: partialValues } },
      { binding: 2, resource: params },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [options.numberOfHeads],
  );
}

function encodeSiluMultiply(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    multiplier: GPUBuffer;
    output: GPUBuffer;
    length: number;
  },
): void {
  const params = paramsBuffer(context, runtime, new Uint32Array([options.length, 0, 0, 0]));
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2SiluShader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: { buffer: options.multiplier } },
      { binding: 2, resource: params },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function encodeF32ToF16(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    input: GPUBuffer;
    output: GPUBuffer;
    inputOffset?: number;
    outputOffset?: number;
    length: number;
  },
): void {
  const params = paramsBuffer(
    context,
    runtime,
    new Uint32Array([
      options.length,
      options.inputOffset ?? 0,
      options.outputOffset ?? 0,
      0,
    ]),
  );
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2F32ToF16Shader,
    [
      { binding: 0, resource: { buffer: options.input } },
      { binding: 1, resource: params },
      { binding: 2, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function encodeResidualAdd(
  context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    left: GPUBuffer;
    right: GPUBuffer;
    output: GPUBuffer;
    length: number;
  },
): void {
  const params = paramsBuffer(context, runtime, new Uint32Array([options.length, 0, 0, 0]));
  encodeComputeShader(
    runtime,
    context.encoder,
    qwen2ResidualAddShader,
    [
      { binding: 0, resource: { buffer: options.left } },
      { binding: 1, resource: { buffer: options.right } },
      { binding: 2, resource: params },
      { binding: 3, resource: { buffer: options.output } },
    ],
    [Math.ceil(options.length / 128)],
  );
}

function validateQkvProjectionShapes(options: {
  qWeight: QwenTensorView;
  kWeight: QwenTensorView;
  vWeight: QwenTensorView;
  inputSize: number;
  qOutputSize: number;
  keyValueOutputSize: number;
}): void {
  const [qOutputSize, qInputSize] = requireMatrixShape(options.qWeight, "qWeight");
  const [kOutputSize, kInputSize] = requireMatrixShape(options.kWeight, "kWeight");
  const [vOutputSize, vInputSize] = requireMatrixShape(options.vWeight, "vWeight");
  if (
    qInputSize !== options.inputSize ||
    kInputSize !== options.inputSize ||
    vInputSize !== options.inputSize ||
    qOutputSize !== options.qOutputSize ||
    kOutputSize !== options.keyValueOutputSize ||
    vOutputSize !== options.keyValueOutputSize
  ) {
    throw new Error(
      `invalid Qwen2 QKV projection shapes: q=[${qOutputSize}, ${qInputSize}], ` +
        `k=[${kOutputSize}, ${kInputSize}], v=[${vOutputSize}, ${vInputSize}]`,
    );
  }
}

function canFuseQwen2QkvProjection(
  qWeight: QwenTensorView,
  kWeight: QwenTensorView,
  vWeight: QwenTensorView,
): boolean {
  const qIsF32 = isFloat32TensorView(qWeight);
  const kIsF32 = isFloat32TensorView(kWeight);
  const vIsF32 = isFloat32TensorView(vWeight);
  if (qIsF32 || kIsF32 || vIsF32) {
    return qIsF32 && kIsF32 && vIsF32;
  }
  return qWeight.type === kWeight.type && qWeight.type === vWeight.type;
}

function qkvBiasBuffer(
  _context: ResidentQwen2GpuContext,
  runtime: WebGpuRuntime,
  options: {
    qBias?: TensorView | Float32Array;
    kBias?: TensorView | Float32Array;
    vBias?: TensorView | Float32Array;
    qOutputSize: number;
    keyValueOutputSize: number;
  },
): GPUBuffer {
  const cacheKey = options.qBias && typeof options.qBias === "object" ? options.qBias : undefined;
  const cached = cacheKey ? qkvBiasDataByQueryBias.get(cacheKey) : undefined;
  if (cached) {
    return createStaticStorageBuffer(runtime, cached);
  }
  const bias = new Float32Array(options.qOutputSize + 2 * options.keyValueOutputSize);
  copyBiasValues(options.qBias, bias, 0, options.qOutputSize, "qBias");
  copyBiasValues(
    options.kBias,
    bias,
    options.qOutputSize,
    options.keyValueOutputSize,
    "kBias",
  );
  copyBiasValues(
    options.vBias,
    bias,
    options.qOutputSize + options.keyValueOutputSize,
    options.keyValueOutputSize,
    "vBias",
  );
  if (cacheKey) {
    qkvBiasDataByQueryBias.set(cacheKey, bias);
  }
  return createStaticStorageBuffer(runtime, bias);
}

function copyBiasValues(
  bias: TensorView | Float32Array | undefined,
  target: Float32Array,
  targetOffset: number,
  expectedLength: number,
  name: string,
): void {
  if (!bias) {
    return;
  }
  const values = bias instanceof Float32Array ? bias : bias.data;
  if (values.length !== expectedLength) {
    throw new Error(`${name} length is ${values.length}, expected ${expectedLength}`);
  }
  target.set(values, targetOffset);
}

function requireMatrixShape(tensor: QwenTensorView, name: string): [number, number] {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }
  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows <= 0 || columns <= 0) {
    throw new Error(`${name} dimensions must be positive, got [${rows}, ${columns}]`);
  }
  return [rows, columns];
}

function quantizedRowByteLength(
  tensor: Exclude<QwenTensorView, TensorView>,
  inputSize: number,
): number {
  if (!Number.isInteger(inputSize) || inputSize <= 0 || inputSize % 32 !== 0) {
    throw new Error(`${tensor.type} length must be a positive multiple of 32, got ${inputSize}`);
  }
  return tensor.type === "q4_0" ? (inputSize / 32) * 18 : (inputSize / 32) * 34;
}

function quantizedTypeCode(tensor: Exclude<QwenTensorView, TensorView>): number {
  return tensor.type === "q4_0" ? 4 : 8;
}

const qwen2ArgmaxShader = `
struct Params {
  vocabularySize: u32,
  topK: u32,
  candidateCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> selectedIds: array<u32>;
@group(0) @binding(3) var<storage, read_write> selectedLogits: array<f32>;
var<workgroup> bestIds: array<u32, 256>;
var<workgroup> bestLogits: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  var bestId = 0u;
  var bestLogit = -3.4028234663852886e38;
  for (var tokenId = lane; tokenId < params.vocabularySize; tokenId = tokenId + 256u) {
    let value = logits[tokenId];
    if (value > bestLogit || (value == bestLogit && tokenId < bestId)) {
      bestId = tokenId;
      bestLogit = value;
    }
  }
  bestIds[lane] = bestId;
  bestLogits[lane] = bestLogit;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let other = lane + stride;
      if (
        bestLogits[other] > bestLogits[lane] ||
        (bestLogits[other] == bestLogits[lane] && bestIds[other] < bestIds[lane])
      ) {
        bestIds[lane] = bestIds[other];
        bestLogits[lane] = bestLogits[other];
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    selectedIds[0] = bestIds[0];
    selectedLogits[0] = bestLogits[0];
  }
}
`;

const qwen2TopKLocalShader = `
struct Params {
  vocabularySize: u32,
  topK: u32,
  candidateCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> candidateIds: array<u32>;
@group(0) @binding(3) var<storage, read_write> candidateLogits: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  var bestIds: array<u32, 64>;
  var bestLogits: array<f32, 64>;
  for (var rank = 0u; rank < params.topK; rank = rank + 1u) {
    bestIds[rank] = 0u;
    bestLogits[rank] = -3.4028234663852886e38;
  }

  for (var tokenId = lane; tokenId < params.vocabularySize; tokenId = tokenId + 256u) {
    let value = logits[tokenId];
    var insertion = params.topK;
    for (var rank = 0u; rank < params.topK; rank = rank + 1u) {
      if (value > bestLogits[rank]) {
        insertion = rank;
        break;
      }
    }
    if (insertion < params.topK) {
      var rank = params.topK - 1u;
      loop {
        if (rank <= insertion) {
          break;
        }
        bestIds[rank] = bestIds[rank - 1u];
        bestLogits[rank] = bestLogits[rank - 1u];
        rank = rank - 1u;
      }
      bestIds[insertion] = tokenId;
      bestLogits[insertion] = value;
    }
  }

  for (var rank = 0u; rank < params.topK; rank = rank + 1u) {
    let outputIndex = lane * params.topK + rank;
    candidateIds[outputIndex] = bestIds[rank];
    candidateLogits[outputIndex] = bestLogits[rank];
  }
}
`;

const qwen2TopKMergeShader = `
struct Params {
  vocabularySize: u32,
  topK: u32,
  candidateCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> candidateIds: array<u32>;
@group(0) @binding(1) var<storage, read> candidateLogits: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> selectedIds: array<u32>;
@group(0) @binding(4) var<storage, read_write> selectedLogits: array<f32>;

var<workgroup> laneRanks: array<u32, 256>;
var<workgroup> reductionIds: array<u32, 256>;
var<workgroup> reductionLogits: array<f32, 256>;
var<workgroup> reductionLanes: array<u32, 256>;

fn better(leftLogit: f32, leftId: u32, rightLogit: f32, rightId: u32) -> bool {
  return leftLogit > rightLogit || (leftLogit == rightLogit && leftId < rightId);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = localId.x;
  laneRanks[lane] = 0u;
  workgroupBarrier();

  for (var outputRank = 0u; outputRank < params.topK; outputRank = outputRank + 1u) {
    let candidateIndex = lane * params.topK + laneRanks[lane];
    reductionIds[lane] = candidateIds[candidateIndex];
    reductionLogits[lane] = candidateLogits[candidateIndex];
    reductionLanes[lane] = lane;
    workgroupBarrier();

    var stride = 128u;
    loop {
      if (lane < stride) {
        let other = lane + stride;
        if (better(
          reductionLogits[other],
          reductionIds[other],
          reductionLogits[lane],
          reductionIds[lane],
        )) {
          reductionIds[lane] = reductionIds[other];
          reductionLogits[lane] = reductionLogits[other];
          reductionLanes[lane] = reductionLanes[other];
        }
      }
      workgroupBarrier();
      if (stride == 1u) {
        break;
      }
      stride = stride / 2u;
    }

    if (lane == 0u) {
      selectedIds[outputRank] = reductionIds[0];
      selectedLogits[outputRank] = reductionLogits[0];
      let winningLane = reductionLanes[0];
      laneRanks[winningLane] = laneRanks[winningLane] + 1u;
    }
    workgroupBarrier();
  }
}
`;

const qwen2F32EmbeddingSequenceShader = `
struct Params {
  sequenceLength: u32,
  embeddingSize: u32,
  entryCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(1) var<storage, read> embedding: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let elementCount = params.sequenceLength * params.embeddingSize;
  if (index >= elementCount) {
    return;
  }
  let position = index / params.embeddingSize;
  let dimension = index % params.embeddingSize;
  let tokenId = tokenIds[position];
  output[index] = embedding[tokenId * params.embeddingSize + dimension];
}
`;

const qwen2QuantizedEmbeddingSequenceShader = `
struct Params {
  sequenceLength: u32,
  embeddingSize: u32,
  rowByteLength: u32,
  quantType: u32,
  entryCount: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> tokenIds: array<u32>;
@group(0) @binding(1) var<storage, read> words: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn byteAt(byteOffset: u32) -> u32 {
  let word = words[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByteAt(byteOffset: u32) -> f32 {
  let value = byteAt(byteOffset);
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let elementCount = params.sequenceLength * params.embeddingSize;
  if (index >= elementCount) {
    return;
  }

  let position = index / params.embeddingSize;
  let dimension = index % params.embeddingSize;
  let tokenId = tokenIds[position];
  let rowOffset = tokenId * params.rowByteLength;
  let block = dimension / 32u;
  let blockIndex = dimension % 32u;
  let blockOffset = rowOffset + block * select(18u, 34u, params.quantType == 8u);
  let scale = f16ToF32(byteAt(blockOffset), byteAt(blockOffset + 1u));

  if (params.quantType == 4u) {
    let packed = byteAt(blockOffset + 2u + (blockIndex % 16u));
    let quantized = select((packed >> 4u) & 0x0fu, packed & 0x0fu, blockIndex < 16u);
    output[index] = f32(i32(quantized) - 8) * scale;
  } else {
    output[index] = signedByteAt(blockOffset + 2u + blockIndex) * scale;
  }
}
`;

const qwen2F32MatrixMultiplySequenceShader = `
struct Params {
  inputSize: u32,
  outputSize: u32,
  sequenceLength: u32,
  inputBaseOffset: u32,
  hasBias: u32,
  _padding0: u32,
  _padding1: u32,
  outputBaseOffset: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let outputElements = params.sequenceLength * params.outputSize;
  if (index >= outputElements) {
    return;
  }
  let position = index / params.outputSize;
  let row = index % params.outputSize;
  var sum = 0.0;
  if (params.hasBias == 1u) {
    sum = bias[row];
  }
  let weightOffset = row * params.inputSize;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;
  for (var column = 0u; column < params.inputSize; column = column + 1u) {
    sum = sum + weight[weightOffset + column] * input[inputOffset + column];
  }
  output[params.outputBaseOffset + index] = sum;
}
`;

const qwen2F32QkvProjectionSequenceShader = `
struct Params {
  inputSize: u32,
  qOutputSize: u32,
  keyValueOutputSize: u32,
  sequenceLength: u32,
  inputBaseOffset: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> qWeight: array<f32>;
@group(0) @binding(1) var<storage, read> kWeight: array<f32>;
@group(0) @binding(2) var<storage, read> vWeight: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> qOutput: array<f32>;
@group(0) @binding(7) var<storage, read_write> kOutput: array<f32>;
@group(0) @binding(8) var<storage, read_write> vOutput: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let outputSize = params.qOutputSize + 2u * params.keyValueOutputSize;
  let outputElements = params.sequenceLength * outputSize;
  if (index >= outputElements) {
    return;
  }

  let position = index / outputSize;
  let projectionRow = index % outputSize;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;

  if (projectionRow < params.qOutputSize) {
    var sum = bias[projectionRow];
    let weightOffset = projectionRow * params.inputSize;
    for (var column = 0u; column < params.inputSize; column = column + 1u) {
      sum = sum + qWeight[weightOffset + column] * input[inputOffset + column];
    }
    qOutput[position * params.qOutputSize + projectionRow] = sum;
    return;
  }

  if (projectionRow < params.qOutputSize + params.keyValueOutputSize) {
    let row = projectionRow - params.qOutputSize;
    var sum = bias[params.qOutputSize + row];
    let weightOffset = row * params.inputSize;
    for (var column = 0u; column < params.inputSize; column = column + 1u) {
      sum = sum + kWeight[weightOffset + column] * input[inputOffset + column];
    }
    kOutput[position * params.keyValueOutputSize + row] = sum;
    return;
  }

  let row = projectionRow - params.qOutputSize - params.keyValueOutputSize;
  var sum = bias[params.qOutputSize + params.keyValueOutputSize + row];
  let weightOffset = row * params.inputSize;
  for (var column = 0u; column < params.inputSize; column = column + 1u) {
    sum = sum + vWeight[weightOffset + column] * input[inputOffset + column];
  }
  vOutput[position * params.keyValueOutputSize + row] = sum;
}
`;

const qwen2QuantizedQkvProjectionSequenceShader = `
struct Params {
  inputSize: u32,
  qOutputSize: u32,
  keyValueOutputSize: u32,
  sequenceLength: u32,
  inputBaseOffset: u32,
  rowByteLength: u32,
  quantType: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> qWeightWords: array<u32>;
@group(0) @binding(1) var<storage, read> kWeightWords: array<u32>;
@group(0) @binding(2) var<storage, read> vWeightWords: array<u32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> qOutput: array<f32>;
@group(0) @binding(7) var<storage, read_write> kOutput: array<f32>;
@group(0) @binding(8) var<storage, read_write> vOutput: array<f32>;

fn qByteAt(byteOffset: u32) -> u32 {
  let word = qWeightWords[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn kByteAt(byteOffset: u32) -> u32 {
  let word = kWeightWords[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn vByteAt(byteOffset: u32) -> u32 {
  let word = vWeightWords[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByte(value: u32) -> f32 {
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

fn qDot(row: u32, inputOffset: u32) -> f32 {
  var sum = 0.0;
  var sourceOffset = row * params.rowByteLength;
  for (var base = 0u; base < params.inputSize; base = base + 32u) {
    let scale = f16ToF32(qByteAt(sourceOffset), qByteAt(sourceOffset + 1u));
    sourceOffset = sourceOffset + 2u;

    if (params.quantType == 4u) {
      for (var packedIndex = 0u; packedIndex < 16u; packedIndex = packedIndex + 1u) {
        let packed = qByteAt(sourceOffset + packedIndex);
        let low = f32(i32(packed & 0x0fu) - 8) * scale;
        let high = f32(i32((packed >> 4u) & 0x0fu) - 8) * scale;
        sum = sum + low * input[inputOffset + base + packedIndex];
        sum = sum + high * input[inputOffset + base + 16u + packedIndex];
      }
      sourceOffset = sourceOffset + 16u;
    } else {
      for (var element = 0u; element < 32u; element = element + 1u) {
        sum = sum + signedByte(qByteAt(sourceOffset + element)) * scale * input[inputOffset + base + element];
      }
      sourceOffset = sourceOffset + 32u;
    }
  }
  return sum;
}

fn kDot(row: u32, inputOffset: u32) -> f32 {
  var sum = 0.0;
  var sourceOffset = row * params.rowByteLength;
  for (var base = 0u; base < params.inputSize; base = base + 32u) {
    let scale = f16ToF32(kByteAt(sourceOffset), kByteAt(sourceOffset + 1u));
    sourceOffset = sourceOffset + 2u;

    if (params.quantType == 4u) {
      for (var packedIndex = 0u; packedIndex < 16u; packedIndex = packedIndex + 1u) {
        let packed = kByteAt(sourceOffset + packedIndex);
        let low = f32(i32(packed & 0x0fu) - 8) * scale;
        let high = f32(i32((packed >> 4u) & 0x0fu) - 8) * scale;
        sum = sum + low * input[inputOffset + base + packedIndex];
        sum = sum + high * input[inputOffset + base + 16u + packedIndex];
      }
      sourceOffset = sourceOffset + 16u;
    } else {
      for (var element = 0u; element < 32u; element = element + 1u) {
        sum = sum + signedByte(kByteAt(sourceOffset + element)) * scale * input[inputOffset + base + element];
      }
      sourceOffset = sourceOffset + 32u;
    }
  }
  return sum;
}

fn vDot(row: u32, inputOffset: u32) -> f32 {
  var sum = 0.0;
  var sourceOffset = row * params.rowByteLength;
  for (var base = 0u; base < params.inputSize; base = base + 32u) {
    let scale = f16ToF32(vByteAt(sourceOffset), vByteAt(sourceOffset + 1u));
    sourceOffset = sourceOffset + 2u;

    if (params.quantType == 4u) {
      for (var packedIndex = 0u; packedIndex < 16u; packedIndex = packedIndex + 1u) {
        let packed = vByteAt(sourceOffset + packedIndex);
        let low = f32(i32(packed & 0x0fu) - 8) * scale;
        let high = f32(i32((packed >> 4u) & 0x0fu) - 8) * scale;
        sum = sum + low * input[inputOffset + base + packedIndex];
        sum = sum + high * input[inputOffset + base + 16u + packedIndex];
      }
      sourceOffset = sourceOffset + 16u;
    } else {
      for (var element = 0u; element < 32u; element = element + 1u) {
        sum = sum + signedByte(vByteAt(sourceOffset + element)) * scale * input[inputOffset + base + element];
      }
      sourceOffset = sourceOffset + 32u;
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let outputSize = params.qOutputSize + 2u * params.keyValueOutputSize;
  let outputElements = params.sequenceLength * outputSize;
  if (index >= outputElements) {
    return;
  }

  let position = index / outputSize;
  let projectionRow = index % outputSize;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;

  if (projectionRow < params.qOutputSize) {
    qOutput[position * params.qOutputSize + projectionRow] =
      bias[projectionRow] + qDot(projectionRow, inputOffset);
    return;
  }

  if (projectionRow < params.qOutputSize + params.keyValueOutputSize) {
    let row = projectionRow - params.qOutputSize;
    kOutput[position * params.keyValueOutputSize + row] =
      bias[params.qOutputSize + row] + kDot(row, inputOffset);
    return;
  }

  let row = projectionRow - params.qOutputSize - params.keyValueOutputSize;
  vOutput[position * params.keyValueOutputSize + row] =
    bias[params.qOutputSize + params.keyValueOutputSize + row] + vDot(row, inputOffset);
}
`;

const qwen2QuantizedMatrixMultiplySequenceShader = `
struct Params {
  inputSize: u32,
  outputSize: u32,
  sequenceLength: u32,
  inputBaseOffset: u32,
  rowByteLength: u32,
  quantType: u32,
  hasBias: u32,
  outputBaseOffset: u32,
}

@group(0) @binding(0) var<storage, read> weightWords: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn byteAt(byteOffset: u32) -> u32 {
  let word = weightWords[byteOffset / 4u];
  let shift = (byteOffset % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn signedByteAt(byteOffset: u32) -> f32 {
  let value = byteAt(byteOffset);
  if (value > 127u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn f16ToF32(lowByte: u32, highByte: u32) -> f32 {
  let half = lowByte | (highByte << 8u);
  let sign = select(1.0, -1.0, (half & 0x8000u) != 0u);
  let exponent = (half >> 10u) & 0x1fu;
  let fraction = half & 0x03ffu;

  if (exponent == 0u) {
    if (fraction == 0u) {
      return sign * 0.0;
    }
    return sign * exp2(-14.0) * (f32(fraction) / 1024.0);
  }
  if (exponent == 0x1fu) {
    return sign * 3.4028234663852886e38;
  }
  return sign * exp2(f32(i32(exponent) - 15)) * (1.0 + f32(fraction) / 1024.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  let outputElements = params.sequenceLength * params.outputSize;
  if (index >= outputElements) {
    return;
  }

  let position = index / params.outputSize;
  let row = index % params.outputSize;
  var sum = 0.0;
  if (params.hasBias == 1u) {
    sum = bias[row];
  }
  var sourceOffset = row * params.rowByteLength;
  let inputOffset = params.inputBaseOffset + position * params.inputSize;
  for (var base = 0u; base < params.inputSize; base = base + 32u) {
    let scale = f16ToF32(byteAt(sourceOffset), byteAt(sourceOffset + 1u));
    sourceOffset = sourceOffset + 2u;

    if (params.quantType == 4u) {
      for (var packedIndex = 0u; packedIndex < 16u; packedIndex = packedIndex + 1u) {
        let packed = byteAt(sourceOffset + packedIndex);
        let low = f32(i32(packed & 0x0fu) - 8) * scale;
        let high = f32(i32((packed >> 4u) & 0x0fu) - 8) * scale;
        sum = sum + low * input[inputOffset + base + packedIndex];
        sum = sum + high * input[inputOffset + base + 16u + packedIndex];
      }
      sourceOffset = sourceOffset + 16u;
    } else {
      for (var element = 0u; element < 32u; element = element + 1u) {
        sum = sum + signedByteAt(sourceOffset + element) * scale * input[inputOffset + base + element];
      }
      sourceOffset = sourceOffset + 32u;
    }
  }
  output[params.outputBaseOffset + index] = sum;
}
`;

const qwen2RmsNormSequenceShader = `
struct Params {
  epsilon: f32,
  inputOffset: f32,
  featureSize: f32,
  sequenceLength: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partialSquares: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let position = workgroupId.x;
  let sequenceLength = u32(params.sequenceLength);
  let featureSize = u32(params.featureSize);
  if (position >= sequenceLength) {
    return;
  }

  let inputBase = u32(params.inputOffset) + position * featureSize;
  let outputBase = position * featureSize;
  let lane = localId.x;
  var localSquare = 0.0;
  for (var index = lane; index < featureSize; index = index + 128u) {
    let value = input[inputBase + index];
    localSquare = localSquare + value * value;
  }
  partialSquares[lane] = localSquare;
  workgroupBarrier();

  for (var stride = 64u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partialSquares[lane] = partialSquares[lane] + partialSquares[lane + stride];
    }
    workgroupBarrier();
  }

  let meanSquare = partialSquares[0] / f32(featureSize);
  let scale = inverseSqrt(meanSquare + params.epsilon);

  for (var index = lane; index < featureSize; index = index + 128u) {
    output[outputBase + index] = input[inputBase + index] * scale * weight[index];
  }
}
`;

const qwen2RoPESequenceShader = `
struct Params {
  theta: f32,
  sequenceLength: f32,
  hiddenSize: f32,
  numberOfHeads: f32,
  headDimension: f32,
  basePosition: f32,
  valuesBaseOffset: f32,
  _padding1: f32,
}

@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> ropeTable: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let pairGlobalIndex = globalId.x;
  let sequenceLength = u32(params.sequenceLength);
  let hiddenSize = u32(params.hiddenSize);
  let numberOfHeads = u32(params.numberOfHeads);
  let headDimension = u32(params.headDimension);
  let halfDimension = headDimension / 2u;
  let pairsPerPosition = numberOfHeads * halfDimension;
  let pairCount = sequenceLength * pairsPerPosition;
  if (pairGlobalIndex >= pairCount) {
    return;
  }

  let position = pairGlobalIndex / pairsPerPosition;
  let headPairIndex = pairGlobalIndex % pairsPerPosition;
  let head = headPairIndex / halfDimension;
  let pairIndex = headPairIndex % halfDimension;
  let base = u32(params.valuesBaseOffset) + position * hiddenSize + head * headDimension;
  let firstIndex = base + pairIndex;
  let secondIndex = base + halfDimension + pairIndex;
  let absolutePosition = params.basePosition + f32(position);
  let tableIndex = (u32(absolutePosition) * halfDimension + pairIndex) * 2u;
  let cosine = ropeTable[tableIndex];
  let sine = ropeTable[tableIndex + 1u];
  let firstValue = values[firstIndex];
  let secondValue = values[secondIndex];
  values[firstIndex] = firstValue * cosine - secondValue * sine;
  values[secondIndex] = secondValue * cosine + firstValue * sine;
}
`;

const qwen2CausalGqaAttentionShader = `
struct Params {
  sequenceLength: u32,
  hiddenSize: u32,
  numberOfHeads: u32,
  numberOfKeyValueHeads: u32,
  headDimension: u32,
  keyValueHiddenSize: u32,
  basePosition: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
var<workgroup> dotProducts: array<f32, 64>;
var<workgroup> softmaxState: array<f32, 4>;

fn kvHiddenIndex(position: u32, keyValueHead: u32, dimension: u32) -> u32 {
  return position * params.keyValueHiddenSize + keyValueHead * params.headDimension + dimension;
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let dimension = localId.x;
  let queryPosition = workgroupId.x;
  let head = workgroupId.y;
  if (queryPosition >= params.sequenceLength || head >= params.numberOfHeads) {
    return;
  }
  let absoluteQueryPosition = params.basePosition + queryPosition;
  let groupSize = params.numberOfHeads / params.numberOfKeyValueHeads;
  let keyValueHead = head / groupSize;
  let scale = inverseSqrt(f32(params.headDimension));
  let queryIndex = queryPosition * params.hiddenSize + head * params.headDimension + dimension;

  var numerator = 0.0;
  if (dimension == 0u) {
    softmaxState[0] = -3.4028234663852886e38;
    softmaxState[1] = 0.0;
  }
  workgroupBarrier();

  for (var keyPosition = 0u; keyPosition <= absoluteQueryPosition; keyPosition = keyPosition + 1u) {
    dotProducts[dimension] = q[queryIndex] * k[kvHiddenIndex(keyPosition, keyValueHead, dimension)];
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
      if (dimension < stride) {
        dotProducts[dimension] = dotProducts[dimension] + dotProducts[dimension + stride];
      }
      workgroupBarrier();
    }

    if (dimension == 0u) {
      let score = dotProducts[0] * scale;
      let nextMaximum = max(softmaxState[0], score);
      let rescale = exp(softmaxState[0] - nextMaximum);
      let weight = exp(score - nextMaximum);
      softmaxState[0] = nextMaximum;
      softmaxState[1] = softmaxState[1] * rescale + weight;
      softmaxState[2] = rescale;
      softmaxState[3] = weight;
    }
    workgroupBarrier();
    numerator = numerator * softmaxState[2] +
      softmaxState[3] * v[kvHiddenIndex(keyPosition, keyValueHead, dimension)];
    workgroupBarrier();
  }

  output[queryIndex] = numerator / softmaxState[1];
}
`;

const qwen2CausalGqaAttentionDecodeShader = `
struct Params {
  position: u32,
  numberOfHeads: u32,
  numberOfKeyValueHeads: u32,
  headDimension: u32,
  keyValueHiddenSize: u32,
  tileSize: u32,
  tileCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> partialStats: array<f32>;
@group(0) @binding(5) var<storage, read_write> partialValues: array<f32>;
var<workgroup> dotProducts: array<f32, 64>;
var<workgroup> softmaxState: array<f32, 4>;

fn qHiddenIndex(head: u32, dimension: u32) -> u32 {
  return head * params.headDimension + dimension;
}

fn kvHiddenIndex(position: u32, keyValueHead: u32, dimension: u32) -> u32 {
  return position * params.keyValueHiddenSize + keyValueHead * params.headDimension + dimension;
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let dimension = localId.x;
  let tile = workgroupId.x;
  let head = workgroupId.y;
  let groupSize = params.numberOfHeads / params.numberOfKeyValueHeads;
  let keyValueHead = head / groupSize;
  let scale = inverseSqrt(f32(params.headDimension));
  let tileStart = tile * params.tileSize;
  let tileEnd = min(tileStart + params.tileSize, params.position + 1u);
  var maximum = -3.4028234663852886e38;
  var denominator = 0.0;
  var weightedValue = 0.0;

  for (var keyPosition = tileStart; keyPosition < tileEnd; keyPosition = keyPosition + 1u) {
    dotProducts[dimension] =
      q[qHiddenIndex(head, dimension)] *
      k[kvHiddenIndex(keyPosition, keyValueHead, dimension)];
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
      if (dimension < stride) {
        dotProducts[dimension] = dotProducts[dimension] + dotProducts[dimension + stride];
      }
      workgroupBarrier();
    }
    if (dimension == 0u) {
      let score = dotProducts[0] * scale;
      let nextMaximum = max(maximum, score);
      let previousScale = exp(maximum - nextMaximum);
      let valueScale = exp(score - nextMaximum);
      maximum = nextMaximum;
      denominator = denominator * previousScale + valueScale;
      softmaxState[0] = maximum;
      softmaxState[1] = denominator;
      softmaxState[2] = previousScale;
      softmaxState[3] = valueScale;
    }
    workgroupBarrier();
    maximum = softmaxState[0];
    denominator = softmaxState[1];
    weightedValue =
      weightedValue * softmaxState[2] +
      softmaxState[3] * v[kvHiddenIndex(keyPosition, keyValueHead, dimension)];
    workgroupBarrier();
  }

  let partialIndex = head * params.tileCount + tile;
  if (dimension == 0u) {
    partialStats[partialIndex * 2u] = maximum;
    partialStats[partialIndex * 2u + 1u] = denominator;
  }
  partialValues[partialIndex * params.headDimension + dimension] = weightedValue;
}
`;

const qwen2CausalGqaAttentionDecodeMergeShader = `
struct Params {
  position: u32,
  numberOfHeads: u32,
  numberOfKeyValueHeads: u32,
  headDimension: u32,
  keyValueHiddenSize: u32,
  tileSize: u32,
  tileCount: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> partialStats: array<f32>;
@group(0) @binding(1) var<storage, read> partialValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> mergedState: array<f32, 2>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let dimension = localId.x;
  let head = workgroupId.x;
  if (dimension == 0u) {
    var maximum = -3.4028234663852886e38;
    for (var tile = 0u; tile < params.tileCount; tile = tile + 1u) {
      maximum = max(maximum, partialStats[(head * params.tileCount + tile) * 2u]);
    }
    var denominator = 0.0;
    for (var tile = 0u; tile < params.tileCount; tile = tile + 1u) {
      let partialIndex = head * params.tileCount + tile;
      denominator = denominator +
        partialStats[partialIndex * 2u + 1u] *
        exp(partialStats[partialIndex * 2u] - maximum);
    }
    mergedState[0] = maximum;
    mergedState[1] = denominator;
  }
  workgroupBarrier();

  var numerator = 0.0;
  for (var tile = 0u; tile < params.tileCount; tile = tile + 1u) {
    let partialIndex = head * params.tileCount + tile;
    numerator = numerator +
      partialValues[partialIndex * params.headDimension + dimension] *
      exp(partialStats[partialIndex * 2u] - mergedState[0]);
  }
  output[head * params.headDimension + dimension] = numerator / mergedState[1];
}
`;

const qwen2F32ToF16Shader = `
enable f16;

struct Params {
  length: u32,
  inputOffset: u32,
  outputOffset: u32,
  _padding0: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index < params.length) {
    output[params.outputOffset + index] = f16(input[params.inputOffset + index]);
  }
}
`;

const qwen2SiluShader = `
struct Params {
  length: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> multiplier: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  let value = input[index];
  output[index] = (value / (1.0 + exp(-value))) * multiplier[index];
}
`;

const qwen2ResidualAddShader = `
struct Params {
  length: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= params.length) {
    return;
  }
  output[index] = left[index] + right[index];
}
`;

const qwen2CausalGqaAttentionF16Shader = withF16KvStorage(qwen2CausalGqaAttentionShader);
const qwen2CausalGqaAttentionDecodeF16Shader = withF16KvStorage(
  qwen2CausalGqaAttentionDecodeShader,
);

function withF16KvStorage(shader: string): string {
  return `enable f16;\n${shader}`
    .replace("var<storage, read> k: array<f32>;", "var<storage, read> k: array<f16>;")
    .replace("var<storage, read> v: array<f32>;", "var<storage, read> v: array<f16>;")
    .replace(/k\[kvHiddenIndex\(([^\n]+)\)\]/g, "f32(k[kvHiddenIndex($1)])")
    .replace(/v\[kvHiddenIndex\(([^\n]+)\)\]/g, "f32(v[kvHiddenIndex($1)])");
}

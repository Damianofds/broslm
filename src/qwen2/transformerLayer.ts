import {
  qwen2AttentionWeights,
  qwen2SelfAttention,
  qwen2SelfAttentionGpu,
  qwen2SelfAttentionIncremental,
  qwen2SelfAttentionIncrementalGpu,
  qwen2SelfAttentionPrefill,
  qwen2SelfAttentionPrefillGpu,
  type Qwen2SelfAttentionConfig,
} from "./attention";
import type { Qwen2LayerKvCache } from "./attentionCache";
import type { Qwen2NormWeights, Qwen2TransformerLayerWeights } from "./loader";
import { qwen2Mlp, qwen2MlpGpu, type Qwen2MlpConfig } from "./mlp";
import { residualAdd, residualAddGpu } from "../primitives/residualAdd";
import { rmsNorm, rmsNormGpu } from "../primitives/rmsNorm";
import type { WebGpuRuntime } from "../runtime/webgpu";

export interface Qwen2TransformerLayerConfig
  extends Qwen2SelfAttentionConfig,
    Qwen2MlpConfig {
  rmsNormEpsilon: number;
}

export function qwen2TransformerLayer(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2TransformerLayerConfig,
  weights: Qwen2TransformerLayerWeights,
): Float32Array {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = new Float32Array(input.length);
  applyRmsNormToSequence(
    input,
    normedForAttention,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.inputLayerNorm,
  );

  const attentionOutput = qwen2SelfAttention(
    normedForAttention,
    sequenceLength,
    config,
    qwen2AttentionWeights(weights.attention),
  );
  const afterAttention = new Float32Array(input.length);
  residualAdd(attentionOutput, input, afterAttention);

  const normedForMlp = new Float32Array(input.length);
  applyRmsNormToSequence(
    afterAttention,
    normedForMlp,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.postAttentionLayerNorm,
  );

  const mlpOutput = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = qwen2Mlp(tokenInput, weights.mlp, config);
    mlpOutput.set(tokenOutput, offset);
  }

  const output = new Float32Array(input.length);
  residualAdd(mlpOutput, afterAttention, output);
  return output;
}

export async function qwen2TransformerLayerGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2TransformerLayerConfig,
  weights: Qwen2TransformerLayerWeights,
): Promise<Float32Array> {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = await applyRmsNormToSequenceGpu(
    runtime,
    input,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.inputLayerNorm,
  );
  const attentionOutput = await qwen2SelfAttentionGpu(
    runtime,
    normedForAttention,
    sequenceLength,
    config,
    qwen2AttentionWeights(weights.attention),
  );
  const afterAttention = await residualAddGpu(runtime, attentionOutput, input);
  const normedForMlp = await applyRmsNormToSequenceGpu(
    runtime,
    afterAttention,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.postAttentionLayerNorm,
  );

  const mlpOutput = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = await qwen2MlpGpu(runtime, tokenInput, weights.mlp, config);
    mlpOutput.set(tokenOutput, offset);
  }

  return residualAddGpu(runtime, mlpOutput, afterAttention);
}

export function qwen2TransformerLayerPrefill(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2TransformerLayerConfig,
  weights: Qwen2TransformerLayerWeights,
  cache: Qwen2LayerKvCache,
): Float32Array {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = new Float32Array(input.length);
  applyRmsNormToSequence(
    input,
    normedForAttention,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.inputLayerNorm,
  );

  const attentionOutput = qwen2SelfAttentionPrefill(
    normedForAttention,
    sequenceLength,
    config,
    qwen2AttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = new Float32Array(input.length);
  residualAdd(attentionOutput, input, afterAttention);

  const normedForMlp = new Float32Array(input.length);
  applyRmsNormToSequence(
    afterAttention,
    normedForMlp,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.postAttentionLayerNorm,
  );

  const mlpOutput = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = qwen2Mlp(tokenInput, weights.mlp, config);
    mlpOutput.set(tokenOutput, offset);
  }

  const output = new Float32Array(input.length);
  residualAdd(mlpOutput, afterAttention, output);
  return output;
}

export async function qwen2TransformerLayerPrefillGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2TransformerLayerConfig,
  weights: Qwen2TransformerLayerWeights,
  cache: Qwen2LayerKvCache,
): Promise<Float32Array> {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = await applyRmsNormToSequenceGpu(
    runtime,
    input,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.inputLayerNorm,
  );
  const attentionOutput = await qwen2SelfAttentionPrefillGpu(
    runtime,
    normedForAttention,
    sequenceLength,
    config,
    qwen2AttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = await residualAddGpu(runtime, attentionOutput, input);
  const normedForMlp = await applyRmsNormToSequenceGpu(
    runtime,
    afterAttention,
    sequenceLength,
    config.hiddenSize,
    config.rmsNormEpsilon,
    weights.postAttentionLayerNorm,
  );

  const mlpOutput = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = await qwen2MlpGpu(runtime, tokenInput, weights.mlp, config);
    mlpOutput.set(tokenOutput, offset);
  }

  return residualAddGpu(runtime, mlpOutput, afterAttention);
}

export function qwen2TransformerLayerIncremental(
  input: Float32Array,
  position: number,
  config: Qwen2TransformerLayerConfig,
  weights: Qwen2TransformerLayerWeights,
  cache: Qwen2LayerKvCache,
): Float32Array {
  validateTransformerTokenInputs(input, position, config);

  const normedForAttention = new Float32Array(config.hiddenSize);
  rmsNorm(input, weights.inputLayerNorm.weight, normedForAttention, {
    featureSize: config.hiddenSize,
    epsilon: config.rmsNormEpsilon,
  });

  const attentionOutput = qwen2SelfAttentionIncremental(
    normedForAttention,
    position,
    config,
    qwen2AttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = new Float32Array(config.hiddenSize);
  residualAdd(attentionOutput, input, afterAttention);

  const normedForMlp = new Float32Array(config.hiddenSize);
  rmsNorm(afterAttention, weights.postAttentionLayerNorm.weight, normedForMlp, {
    featureSize: config.hiddenSize,
    epsilon: config.rmsNormEpsilon,
  });

  const mlpOutput = qwen2Mlp(normedForMlp, weights.mlp, config);
  const output = new Float32Array(config.hiddenSize);
  residualAdd(mlpOutput, afterAttention, output);
  return output;
}

export async function qwen2TransformerLayerIncrementalGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  position: number,
  config: Qwen2TransformerLayerConfig,
  weights: Qwen2TransformerLayerWeights,
  cache: Qwen2LayerKvCache,
): Promise<Float32Array> {
  validateTransformerTokenInputs(input, position, config);

  const normedForAttention = await rmsNormGpu(
    runtime,
    input,
    weights.inputLayerNorm.weight,
    {
      featureSize: config.hiddenSize,
      epsilon: config.rmsNormEpsilon,
    },
  );
  const attentionOutput = await qwen2SelfAttentionIncrementalGpu(
    runtime,
    normedForAttention,
    position,
    config,
    qwen2AttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = await residualAddGpu(runtime, attentionOutput, input);
  const normedForMlp = await rmsNormGpu(
    runtime,
    afterAttention,
    weights.postAttentionLayerNorm.weight,
    {
      featureSize: config.hiddenSize,
      epsilon: config.rmsNormEpsilon,
    },
  );
  const mlpOutput = await qwen2MlpGpu(runtime, normedForMlp, weights.mlp, config);
  return residualAddGpu(runtime, mlpOutput, afterAttention);
}

function applyRmsNormToSequence(
  input: Float32Array,
  output: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  epsilon: number,
  weights: Qwen2NormWeights,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    rmsNorm(input, weights.weight, output, {
      inputOffset: offset,
      outputOffset: offset,
      featureSize: hiddenSize,
      epsilon,
    });
  }
}

async function applyRmsNormToSequenceGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  epsilon: number,
  weights: Qwen2NormWeights,
): Promise<Float32Array> {
  const output = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    const tokenOutput = await rmsNormGpu(runtime, input, weights.weight, {
      inputOffset: offset,
      featureSize: hiddenSize,
      epsilon,
    });
    output.set(tokenOutput, offset);
  }
  return output;
}

function validateTransformerLayerInputs(
  input: Float32Array,
  sequenceLength: number,
  config: Qwen2TransformerLayerConfig,
): void {
  assertPositiveInteger(sequenceLength, "sequenceLength");
  validateCommonConfig(config);

  const expectedLength = sequenceLength * config.hiddenSize;
  if (input.length !== expectedLength) {
    throw new Error(`input length is ${input.length}, expected ${expectedLength}`);
  }
}

function validateTransformerTokenInputs(
  input: Float32Array,
  position: number,
  config: Qwen2TransformerLayerConfig,
): void {
  if (!Number.isInteger(position) || position < 0) {
    throw new RangeError(`position must be a non-negative integer, got ${position}`);
  }
  validateCommonConfig(config);

  if (input.length !== config.hiddenSize) {
    throw new Error(`input length is ${input.length}, expected ${config.hiddenSize}`);
  }
}

function validateCommonConfig(config: Qwen2TransformerLayerConfig): void {
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.intermediateSize, "intermediateSize");
  assertPositiveInteger(config.numberOfHeads, "numberOfHeads");
  assertPositiveInteger(config.numberOfKeyValueHeads, "numberOfKeyValueHeads");
  assertPositiveInteger(config.headDimension, "headDimension");
  assertPositiveInteger(config.keyValueHiddenSize, "keyValueHiddenSize");
  assertPositiveNumber(config.rmsNormEpsilon, "rmsNormEpsilon");
  assertPositiveNumber(config.ropeTheta, "ropeTheta");

  if (config.hiddenSize !== config.numberOfHeads * config.headDimension) {
    throw new Error("Qwen2 hiddenSize must equal numberOfHeads * headDimension");
  }
  if (config.keyValueHiddenSize !== config.numberOfKeyValueHeads * config.headDimension) {
    throw new Error("Qwen2 keyValueHiddenSize must equal numberOfKeyValueHeads * headDimension");
  }
  if (config.numberOfHeads % config.numberOfKeyValueHeads !== 0) {
    throw new Error("Qwen2 numberOfHeads must be divisible by numberOfKeyValueHeads");
  }
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

function assertPositiveNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number, got ${String(value)}`);
  }
}

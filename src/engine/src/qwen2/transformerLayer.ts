import {
  qwen2AttentionWeights,
  qwen2SelfAttention,
  qwen2SelfAttentionIncremental,
  qwen2SelfAttentionPrefill,
  type Qwen2SelfAttentionConfig,
} from "./attention";
import type { Qwen2LayerKvCache } from "./attentionCache";
import type { Qwen2NormWeights, Qwen2TransformerLayerWeights } from "./loader";
import { qwen2Mlp, type Qwen2MlpConfig } from "./mlp";
import { residualAdd } from "../primitives/residualAdd";
import { rmsNorm } from "../primitives/rmsNorm";

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

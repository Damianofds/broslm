import {
  causalSelfAttention,
  causalSelfAttentionGpu,
  causalSelfAttentionIncrementalGpu,
  causalSelfAttentionPrefillGpu,
  causalSelfAttentionIncremental,
  causalSelfAttentionPrefill,
  type CausalSelfAttentionConfig,
  gptNeoAttentionWeights,
} from "./attention";
import type { LayerKvCache } from "./attentionCache";
import type { LayerNormWeights, TransformerLayerWeights } from "./loader";
import { gptNeoMlpWeights, mlp, mlpGpu, type MlpConfig } from "./mlp";
import { layerNorm, layerNormGpu } from "../primitives/layerNorm";
import { residualAdd, residualAddGpu } from "../primitives/residualAdd";
import type { WebGpuRuntime } from "../runtime/webgpu";

export interface TransformerLayerConfig extends CausalSelfAttentionConfig, MlpConfig {
  layerNormEpsilon: number;
}

export function transformerLayer(
  input: Float32Array,
  sequenceLength: number,
  config: TransformerLayerConfig,
  weights: TransformerLayerWeights,
): Float32Array {
  validateTransformerLayerInputs(input, sequenceLength, config);

  // GPT-Neo pre-norm attention block:
  // [sequenceLength, hiddenSize] -> ln_1 -> causal attention -> residual add.
  const normedForAttention = new Float32Array(input.length);
  applyLayerNormToSequence(
    input,
    normedForAttention,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln1,
  );

  const attentionOutput = causalSelfAttention(
    normedForAttention,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
  );
  const afterAttention = new Float32Array(input.length);
  residualAdd(attentionOutput, input, afterAttention);

  // GPT-Neo pre-norm MLP block:
  // [sequenceLength, hiddenSize] -> ln_2 -> per-token MLP -> residual add.
  const normedForMlp = new Float32Array(input.length);
  applyLayerNormToSequence(
    afterAttention,
    normedForMlp,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln2,
  );

  const mlpOutput = new Float32Array(input.length);
  const adaptedMlpWeights = gptNeoMlpWeights(weights.mlp);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = mlp(tokenInput, adaptedMlpWeights, config);
    mlpOutput.set(tokenOutput, offset);
  }

  const output = new Float32Array(input.length);
  residualAdd(mlpOutput, afterAttention, output);
  return output;
}

export async function transformerLayerGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: TransformerLayerConfig,
  weights: TransformerLayerWeights,
): Promise<Float32Array> {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = await applyLayerNormToSequenceGpu(
    runtime,
    input,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln1,
  );
  const attentionOutput = await causalSelfAttentionGpu(
    runtime,
    normedForAttention,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
  );
  const afterAttention = await residualAddGpu(runtime, attentionOutput, input);
  const normedForMlp = await applyLayerNormToSequenceGpu(
    runtime,
    afterAttention,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln2,
  );

  const mlpOutput = new Float32Array(input.length);
  const adaptedMlpWeights = gptNeoMlpWeights(weights.mlp);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = await mlpGpu(runtime, tokenInput, adaptedMlpWeights, config);
    mlpOutput.set(tokenOutput, offset);
  }

  return residualAddGpu(runtime, mlpOutput, afterAttention);
}

export function transformerLayerPrefill(
  input: Float32Array,
  sequenceLength: number,
  config: TransformerLayerConfig,
  weights: TransformerLayerWeights,
  cache: LayerKvCache,
): Float32Array {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = new Float32Array(input.length);
  applyLayerNormToSequence(
    input,
    normedForAttention,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln1,
  );

  const attentionOutput = causalSelfAttentionPrefill(
    normedForAttention,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = new Float32Array(input.length);
  residualAdd(attentionOutput, input, afterAttention);

  const normedForMlp = new Float32Array(input.length);
  applyLayerNormToSequence(
    afterAttention,
    normedForMlp,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln2,
  );

  const mlpOutput = new Float32Array(input.length);
  const adaptedMlpWeights = gptNeoMlpWeights(weights.mlp);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = mlp(tokenInput, adaptedMlpWeights, config);
    mlpOutput.set(tokenOutput, offset);
  }

  const output = new Float32Array(input.length);
  residualAdd(mlpOutput, afterAttention, output);
  return output;
}

export async function transformerLayerPrefillGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  config: TransformerLayerConfig,
  weights: TransformerLayerWeights,
  cache: LayerKvCache,
): Promise<Float32Array> {
  validateTransformerLayerInputs(input, sequenceLength, config);

  const normedForAttention = await applyLayerNormToSequenceGpu(
    runtime,
    input,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln1,
  );
  const attentionOutput = await causalSelfAttentionPrefillGpu(
    runtime,
    normedForAttention,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = await residualAddGpu(runtime, attentionOutput, input);
  const normedForMlp = await applyLayerNormToSequenceGpu(
    runtime,
    afterAttention,
    sequenceLength,
    config.hiddenSize,
    config.layerNormEpsilon,
    weights.ln2,
  );

  const mlpOutput = new Float32Array(input.length);
  const adaptedMlpWeights = gptNeoMlpWeights(weights.mlp);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    const tokenInput = normedForMlp.subarray(offset, offset + config.hiddenSize);
    const tokenOutput = await mlpGpu(runtime, tokenInput, adaptedMlpWeights, config);
    mlpOutput.set(tokenOutput, offset);
  }

  return residualAddGpu(runtime, mlpOutput, afterAttention);
}

export function transformerLayerIncremental(
  input: Float32Array,
  position: number,
  config: TransformerLayerConfig,
  weights: TransformerLayerWeights,
  cache: LayerKvCache,
): Float32Array {
  validateTransformerTokenInputs(input, position, config);

  const normedForAttention = new Float32Array(config.hiddenSize);
  layerNorm(input, weights.ln1.weight, weights.ln1.bias, normedForAttention, {
    featureSize: config.hiddenSize,
    epsilon: config.layerNormEpsilon,
  });

  const attentionOutput = causalSelfAttentionIncremental(
    normedForAttention,
    position,
    config,
    gptNeoAttentionWeights(weights.attention),
    cache,
  );

  const afterAttention = new Float32Array(config.hiddenSize);
  for (let index = 0; index < config.hiddenSize; index += 1) {
    afterAttention[index] = (attentionOutput[index] ?? 0) + (input[index] ?? 0);
  }

  const normedForMlp = new Float32Array(config.hiddenSize);
  layerNorm(afterAttention, weights.ln2.weight, weights.ln2.bias, normedForMlp, {
    featureSize: config.hiddenSize,
    epsilon: config.layerNormEpsilon,
  });

  const mlpOutput = mlp(normedForMlp, gptNeoMlpWeights(weights.mlp), config);
  const output = new Float32Array(config.hiddenSize);
  for (let index = 0; index < config.hiddenSize; index += 1) {
    output[index] = (mlpOutput[index] ?? 0) + (afterAttention[index] ?? 0);
  }

  return output;
}

export async function transformerLayerIncrementalGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  position: number,
  config: TransformerLayerConfig,
  weights: TransformerLayerWeights,
  cache: LayerKvCache,
): Promise<Float32Array> {
  validateTransformerTokenInputs(input, position, config);

  const normedForAttention = await layerNormGpu(
    runtime,
    input,
    weights.ln1.weight,
    weights.ln1.bias,
    {
      featureSize: config.hiddenSize,
      epsilon: config.layerNormEpsilon,
    },
  );
  const attentionOutput = await causalSelfAttentionIncrementalGpu(
    runtime,
    normedForAttention,
    position,
    config,
    gptNeoAttentionWeights(weights.attention),
    cache,
  );
  const afterAttention = await residualAddGpu(runtime, attentionOutput, input);
  const normedForMlp = await layerNormGpu(
    runtime,
    afterAttention,
    weights.ln2.weight,
    weights.ln2.bias,
    {
      featureSize: config.hiddenSize,
      epsilon: config.layerNormEpsilon,
    },
  );
  const mlpOutput = await mlpGpu(runtime, normedForMlp, gptNeoMlpWeights(weights.mlp), config);
  return residualAddGpu(runtime, mlpOutput, afterAttention);
}

function applyLayerNormToSequence(
  input: Float32Array,
  output: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  epsilon: number,
  weights: LayerNormWeights,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    layerNorm(input, weights.weight, weights.bias, output, {
      inputOffset: offset,
      outputOffset: offset,
      featureSize: hiddenSize,
      epsilon,
    });
  }
}

async function applyLayerNormToSequenceGpu(
  runtime: WebGpuRuntime,
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  epsilon: number,
  weights: LayerNormWeights,
): Promise<Float32Array> {
  const output = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    const tokenOutput = await layerNormGpu(runtime, input, weights.weight, weights.bias, {
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
  config: TransformerLayerConfig,
): void {
  assertPositiveInteger(sequenceLength, "sequenceLength");
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.intermediateSize, "intermediateSize");
  assertPositiveInteger(config.numberOfHeads, "numberOfHeads");
  assertPositiveInteger(config.headDimension, "headDimension");

  if (config.hiddenSize % config.numberOfHeads !== 0) {
    throw new Error(
      `hiddenSize ${config.hiddenSize} must be divisible by numberOfHeads ${config.numberOfHeads}`,
    );
  }

  const expectedHeadDimension = config.hiddenSize / config.numberOfHeads;
  if (config.headDimension !== expectedHeadDimension) {
    throw new Error(
      `headDimension ${config.headDimension} must equal hiddenSize / numberOfHeads ${expectedHeadDimension}`,
    );
  }

  if (
    typeof config.layerNormEpsilon !== "number" ||
    !Number.isFinite(config.layerNormEpsilon) ||
    config.layerNormEpsilon <= 0
  ) {
    throw new RangeError(
      `layerNormEpsilon must be a positive finite number, got ${config.layerNormEpsilon}`,
    );
  }

  const expectedLength = sequenceLength * config.hiddenSize;
  if (input.length !== expectedLength) {
    throw new Error(`input length is ${input.length}, expected ${expectedLength}`);
  }
}

function validateTransformerTokenInputs(
  input: Float32Array,
  position: number,
  config: TransformerLayerConfig,
): void {
  assertPositiveInteger(position + 1, "position");
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.intermediateSize, "intermediateSize");
  assertPositiveInteger(config.numberOfHeads, "numberOfHeads");
  assertPositiveInteger(config.headDimension, "headDimension");

  if (input.length !== config.hiddenSize) {
    throw new Error(`input length is ${input.length}, expected ${config.hiddenSize}`);
  }
  if (config.hiddenSize % config.numberOfHeads !== 0) {
    throw new Error(
      `hiddenSize ${config.hiddenSize} must be divisible by numberOfHeads ${config.numberOfHeads}`,
    );
  }

  const expectedHeadDimension = config.hiddenSize / config.numberOfHeads;
  if (config.headDimension !== expectedHeadDimension) {
    throw new Error(
      `headDimension ${config.headDimension} must equal hiddenSize / numberOfHeads ${expectedHeadDimension}`,
    );
  }
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

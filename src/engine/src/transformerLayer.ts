import {
  causalSelfAttention,
  type CausalSelfAttentionConfig,
  gptNeoAttentionWeights,
} from "./attention";
import type { LayerNormWeights, TransformerLayerWeights } from "./loader";
import { gptNeoMlpWeights, mlp, type MlpConfig } from "./mlp";
import { layerNorm } from "./primitives/layerNorm";
import { residualAdd } from "./primitives/residualAdd";

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

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

import type {
  AttentionWeights as BoundAttentionWeights,
  LayerNormWeights,
  TensorView,
} from "./loader";
import { layerNorm } from "./primitives/layerNorm";
import { matrixVectorMultiply } from "./primitives/matrixVectorMultiply";
import { residualAdd } from "./primitives/residualAdd";
import { softmax } from "./primitives/softmax";

export interface CausalSelfAttentionConfig {
  hiddenSize: number;
  numberOfHeads: number;
  headDimension: number;
}

export interface CausalSelfAttentionWeights {
  query: TensorView;
  key: TensorView;
  value: TensorView;
  output: TensorView;
  outputBias?: TensorView | Float32Array;
}

export interface CausalSelfAttentionDebug {
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  rawScores: Float32Array;
  scaledScores: Float32Array;
  probabilities: Float32Array;
  headOutput: Float32Array;
  concatenated: Float32Array;
  projected: Float32Array;
}

export interface CausalSelfAttentionResult {
  output: Float32Array;
  debug: CausalSelfAttentionDebug;
}

export interface TransformerAttentionBlockWeights {
  layerNorm: LayerNormWeights;
  attention: BoundAttentionWeights;
}

export function causalSelfAttention(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): Float32Array {
  return causalSelfAttentionWithDebug(input, sequenceLength, config, weights).output;
}

export function causalSelfAttentionWithDebug(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): CausalSelfAttentionResult {
  validateAttentionInputs(input, sequenceLength, config, weights);

  const { hiddenSize, numberOfHeads, headDimension } = config;
  const hiddenElements = sequenceLength * hiddenSize;
  const scoreElements = numberOfHeads * sequenceLength * sequenceLength;
  const q = new Float32Array(hiddenElements);
  const k = new Float32Array(hiddenElements);
  const v = new Float32Array(hiddenElements);
  const rawScores = new Float32Array(scoreElements);
  const scaledScores = new Float32Array(scoreElements);
  const probabilities = new Float32Array(scoreElements);
  const headOutput = new Float32Array(hiddenElements);
  const concatenated = new Float32Array(hiddenElements);
  const projected = new Float32Array(hiddenElements);

  rawScores.fill(Number.NaN);
  scaledScores.fill(Number.NaN);
  probabilities.fill(Number.NaN);

  projectSequence(input, sequenceLength, hiddenSize, weights.query, q);
  projectSequence(input, sequenceLength, hiddenSize, weights.key, k);
  projectSequence(input, sequenceLength, hiddenSize, weights.value, v);

  computeCausalHeadOutputs({
    q,
    k,
    v,
    rawScores,
    scaledScores,
    probabilities,
    headOutput,
    concatenated,
    sequenceLength,
    hiddenSize,
    numberOfHeads,
    headDimension,
  });

  projectSequence(
    concatenated,
    sequenceLength,
    hiddenSize,
    weights.output,
    projected,
    weights.outputBias,
  );

  return {
    output: projected,
    debug: {
      q,
      k,
      v,
      rawScores,
      scaledScores,
      probabilities,
      headOutput,
      concatenated,
      projected,
    },
  };
}

export function gptNeoAttentionWeights(weights: BoundAttentionWeights): CausalSelfAttentionWeights {
  return {
    query: weights.qProjWeight,
    key: weights.kProjWeight,
    value: weights.vProjWeight,
    output: weights.outProjWeight,
    outputBias: weights.outProjBias,
  };
}

export function transformerAttentionBlock(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig & { layerNormEpsilon: number },
  weights: TransformerAttentionBlockWeights,
): Float32Array {
  validateSequenceInput(input, sequenceLength, config.hiddenSize);

  const normed = new Float32Array(input.length);
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * config.hiddenSize;
    layerNorm(input, weights.layerNorm.weight, weights.layerNorm.bias, normed, {
      inputOffset: offset,
      outputOffset: offset,
      featureSize: config.hiddenSize,
      epsilon: config.layerNormEpsilon,
    });
  }

  const attention = causalSelfAttention(
    normed,
    sequenceLength,
    config,
    gptNeoAttentionWeights(weights.attention),
  );
  const output = new Float32Array(input.length);
  residualAdd(attention, input, output);
  return output;
}

function projectSequence(
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
  weight: TensorView,
  output: Float32Array,
  bias?: TensorView | Float32Array,
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    const offset = position * hiddenSize;
    matrixVectorMultiply(weight, input, output, {
      bias,
      inputOffset: offset,
      outputOffset: offset,
    });
  }
}

function computeCausalHeadOutputs(options: {
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  rawScores: Float32Array;
  scaledScores: Float32Array;
  probabilities: Float32Array;
  headOutput: Float32Array;
  concatenated: Float32Array;
  sequenceLength: number;
  hiddenSize: number;
  numberOfHeads: number;
  headDimension: number;
}): void {
  const {
    q,
    k,
    v,
    rawScores,
    scaledScores,
    probabilities,
    headOutput,
    concatenated,
    sequenceLength,
    hiddenSize,
    numberOfHeads,
    headDimension,
  } = options;
  const scale = 1 / Math.sqrt(headDimension);

  for (let head = 0; head < numberOfHeads; head += 1) {
    for (let queryPosition = 0; queryPosition < sequenceLength; queryPosition += 1) {
      const visibleLength = queryPosition + 1;
      const scoreOffset = scoreIndex(head, queryPosition, 0, sequenceLength);

      for (let keyPosition = 0; keyPosition <= queryPosition; keyPosition += 1) {
        let score = 0;
        for (let dimension = 0; dimension < headDimension; dimension += 1) {
          score +=
            (q[hiddenIndex(queryPosition, head, dimension, hiddenSize, headDimension)] ?? 0) *
            (k[hiddenIndex(keyPosition, head, dimension, hiddenSize, headDimension)] ?? 0);
        }

        const index = scoreOffset + keyPosition;
        rawScores[index] = score;
        scaledScores[index] = score * scale;
      }

      softmax(scaledScores, probabilities, {
        inputOffset: scoreOffset,
        outputOffset: scoreOffset,
        length: visibleLength,
      });

      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        let value = 0;
        for (let valuePosition = 0; valuePosition <= queryPosition; valuePosition += 1) {
          value +=
            (probabilities[scoreOffset + valuePosition] ?? 0) *
            (v[hiddenIndex(valuePosition, head, dimension, hiddenSize, headDimension)] ?? 0);
        }

        const outputIndex = hiddenIndex(queryPosition, head, dimension, hiddenSize, headDimension);
        headOutput[outputIndex] = value;
        concatenated[outputIndex] = value;
      }
    }
  }
}

function validateAttentionInputs(
  input: Float32Array,
  sequenceLength: number,
  config: CausalSelfAttentionConfig,
  weights: CausalSelfAttentionWeights,
): void {
  validateSequenceInput(input, sequenceLength, config.hiddenSize);
  validateAttentionConfig(config);
  requireMatrixShape(weights.query, "query", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.key, "key", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.value, "value", config.hiddenSize, config.hiddenSize);
  requireMatrixShape(weights.output, "output", config.hiddenSize, config.hiddenSize);

  if (weights.outputBias instanceof Float32Array) {
    if (weights.outputBias.length !== config.hiddenSize) {
      throw new Error(`outputBias length is ${weights.outputBias.length}, expected ${config.hiddenSize}`);
    }
  } else if (weights.outputBias) {
    requireVectorShape(weights.outputBias, "outputBias", config.hiddenSize);
  }
}

function validateAttentionConfig(config: CausalSelfAttentionConfig): void {
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
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
}

function validateSequenceInput(
  input: Float32Array,
  sequenceLength: number,
  hiddenSize: number,
): void {
  assertPositiveInteger(sequenceLength, "sequenceLength");
  assertPositiveInteger(hiddenSize, "hiddenSize");

  const expectedLength = sequenceLength * hiddenSize;
  if (input.length !== expectedLength) {
    throw new Error(`input length is ${input.length}, expected ${expectedLength}`);
  }
}

function requireMatrixShape(
  tensor: TensorView,
  name: string,
  expectedRows: number,
  expectedColumns: number,
): void {
  if (tensor.shape.length !== 2) {
    throw new Error(`${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }

  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows !== expectedRows || columns !== expectedColumns) {
    throw new Error(
      `${name} shape is [${tensor.shape.join(", ")}], expected [${expectedRows}, ${expectedColumns}]`,
    );
  }
  if (tensor.data.length !== rows * columns) {
    throw new Error(`${name} data length does not match shape [${rows}, ${columns}]`);
  }
}

function requireVectorShape(tensor: TensorView, name: string, expectedLength: number): void {
  if (tensor.shape.length !== 1 || tensor.shape[0] !== expectedLength) {
    throw new Error(`${name} must be rank 1 with length ${expectedLength}`);
  }
  if (tensor.data.length !== expectedLength) {
    throw new Error(`${name} data length does not match shape [${expectedLength}]`);
  }
}

function hiddenIndex(
  position: number,
  head: number,
  dimension: number,
  hiddenSize: number,
  headDimension: number,
): number {
  return position * hiddenSize + head * headDimension + dimension;
}

function scoreIndex(
  head: number,
  queryPosition: number,
  keyPosition: number,
  sequenceLength: number,
): number {
  return head * sequenceLength * sequenceLength + queryPosition * sequenceLength + keyPosition;
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

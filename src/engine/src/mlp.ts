import type { MlpWeights as BoundMlpWeights, TensorView } from "./loader";
import { gelu } from "./primitives/gelu";
import { matrixVectorMultiply } from "./primitives/matrixVectorMultiply";

export interface MlpConfig {
  hiddenSize: number;
  intermediateSize: number;
}

export interface MlpWeights {
  upWeight: TensorView;
  upBias?: TensorView | Float32Array;
  downWeight: TensorView;
  downBias?: TensorView | Float32Array;
}

export function mlp(
  input: Float32Array,
  weights: MlpWeights,
  config: MlpConfig,
): Float32Array {
  validateMlpInputs(input, weights, config);

  // input: [hiddenSize]
  // upWeight: row-major [intermediateSize, hiddenSize]
  // intermediate: [intermediateSize]
  const intermediate = new Float32Array(config.intermediateSize);
  matrixVectorMultiply(weights.upWeight, input, intermediate, {
    bias: weights.upBias,
  });

  // GELU is applied independently to the expanded hidden features.
  gelu(intermediate, intermediate);

  // downWeight: row-major [hiddenSize, intermediateSize]
  // output: [hiddenSize]
  const output = new Float32Array(config.hiddenSize);
  matrixVectorMultiply(weights.downWeight, intermediate, output, {
    bias: weights.downBias,
  });

  return output;
}

export function gptNeoMlpWeights(weights: BoundMlpWeights): MlpWeights {
  return {
    upWeight: weights.cFcWeight,
    upBias: weights.cFcBias,
    downWeight: weights.cProjWeight,
    downBias: weights.cProjBias,
  };
}

function validateMlpInputs(
  input: Float32Array,
  weights: MlpWeights,
  config: MlpConfig,
): void {
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.intermediateSize, "intermediateSize");

  if (input.length !== config.hiddenSize) {
    throw new Error(
      `Invalid MLP input size: expected ${config.hiddenSize} values, got ${input.length}`,
    );
  }

  requireMatrixShape(
    weights.upWeight,
    "upWeight",
    config.intermediateSize,
    config.hiddenSize,
  );
  requireOptionalBias(weights.upBias, "upBias", config.intermediateSize);
  requireMatrixShape(
    weights.downWeight,
    "downWeight",
    config.hiddenSize,
    config.intermediateSize,
  );
  requireOptionalBias(weights.downBias, "downBias", config.hiddenSize);
}

function requireMatrixShape(
  tensor: TensorView,
  name: string,
  expectedRows: number,
  expectedColumns: number,
): void {
  if (tensor.shape.length !== 2) {
    throw new Error(`MLP ${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }

  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows !== expectedRows || columns !== expectedColumns) {
    throw new Error(
      `Invalid MLP ${name} shape: expected [${expectedRows}, ${expectedColumns}], got ` +
        `[${tensor.shape.join(", ")}]`,
    );
  }

  const expectedLength = expectedRows * expectedColumns;
  if (tensor.data.length !== expectedLength) {
    throw new Error(
      `Invalid MLP ${name} size: expected ${expectedLength} values, got ${tensor.data.length}`,
    );
  }
}

function requireOptionalBias(
  bias: TensorView | Float32Array | undefined,
  name: string,
  expectedLength: number,
): void {
  if (!bias) {
    return;
  }

  const values = bias instanceof Float32Array ? bias : bias.data;
  if (!(values instanceof Float32Array)) {
    throw new Error(`MLP ${name} must resolve to a Float32Array`);
  }

  if (!(bias instanceof Float32Array) && (bias.shape.length !== 1 || bias.shape[0] !== expectedLength)) {
    throw new Error(
      `Invalid MLP ${name} shape: expected [${expectedLength}], got [${bias.shape.join(", ")}]`,
    );
  }

  if (values.length !== expectedLength) {
    throw new Error(
      `Invalid MLP ${name} size: expected ${expectedLength} values, got ${values.length}`,
    );
  }
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

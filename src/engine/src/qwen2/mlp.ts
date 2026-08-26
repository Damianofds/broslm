import type { Qwen2MlpWeights } from "./loader";
import { matrixVectorMultiplyQwen, type QwenTensorView } from "./quantizedTensor";
import { elementwiseMultiply } from "../primitives/elementwiseMultiply";
import { silu } from "../primitives/silu";

export interface Qwen2MlpConfig {
  hiddenSize: number;
  intermediateSize: number;
}

export interface Qwen2MlpDebug {
  gate: Float32Array;
  up: Float32Array;
  activatedGate: Float32Array;
  gated: Float32Array;
}

export interface Qwen2MlpResult {
  output: Float32Array;
  debug: Qwen2MlpDebug;
}

export function qwen2Mlp(
  input: Float32Array,
  weights: Qwen2MlpWeights,
  config: Qwen2MlpConfig,
): Float32Array {
  return qwen2MlpWithDebug(input, weights, config).output;
}

export function qwen2MlpWithDebug(
  input: Float32Array,
  weights: Qwen2MlpWeights,
  config: Qwen2MlpConfig,
): Qwen2MlpResult {
  validateMlpInputs(input, weights, config);

  const gate = new Float32Array(config.intermediateSize);
  const up = new Float32Array(config.intermediateSize);
  const activatedGate = new Float32Array(config.intermediateSize);
  const gated = new Float32Array(config.intermediateSize);
  const output = new Float32Array(config.hiddenSize);

  matrixVectorMultiplyQwen(weights.gateProjWeight, input, gate);
  matrixVectorMultiplyQwen(weights.upProjWeight, input, up);
  silu(gate, activatedGate);
  elementwiseMultiply(activatedGate, up, gated);
  matrixVectorMultiplyQwen(weights.downProjWeight, gated, output);

  return {
    output,
    debug: {
      gate,
      up,
      activatedGate,
      gated,
    },
  };
}

function validateMlpInputs(
  input: Float32Array,
  weights: Qwen2MlpWeights,
  config: Qwen2MlpConfig,
): void {
  assertPositiveInteger(config.hiddenSize, "hiddenSize");
  assertPositiveInteger(config.intermediateSize, "intermediateSize");
  if (input.length !== config.hiddenSize) {
    throw new Error(
      `Invalid Qwen2 MLP input size: expected ${config.hiddenSize} values, got ${input.length}`,
    );
  }

  requireMatrixShape(
    weights.gateProjWeight,
    "gateProjWeight",
    config.intermediateSize,
    config.hiddenSize,
  );
  requireMatrixShape(
    weights.upProjWeight,
    "upProjWeight",
    config.intermediateSize,
    config.hiddenSize,
  );
  requireMatrixShape(
    weights.downProjWeight,
    "downProjWeight",
    config.hiddenSize,
    config.intermediateSize,
  );
}

function requireMatrixShape(
  tensor: QwenTensorView,
  name: string,
  expectedRows: number,
  expectedColumns: number,
): void {
  if (tensor.shape.length !== 2) {
    throw new Error(`Qwen2 MLP ${name} must be rank 2, got shape [${tensor.shape.join(", ")}]`);
  }

  const rows = tensor.shape[0] ?? 0;
  const columns = tensor.shape[1] ?? 0;
  if (rows !== expectedRows || columns !== expectedColumns) {
    throw new Error(
      `Invalid Qwen2 MLP ${name} shape: expected [${expectedRows}, ${expectedColumns}], got ` +
        `[${tensor.shape.join(", ")}]`,
    );
  }
}

function assertPositiveInteger(value: unknown, name: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer, got ${String(value)}`);
  }
}

import { describe, expect, it } from "vitest";
import type { Qwen2SelfAttentionConfig, Qwen2SelfAttentionWeights } from "../../src/qwen2/attention";
import {
  qwen2SelfAttentionIncremental,
  qwen2SelfAttentionWithDebug,
} from "../../src/qwen2/attention";
import type { Qwen2LayerKvCache } from "../../src/qwen2/attentionCache";
import type { TensorView } from "../../src/qwen2/loader";

describe("qwen2SelfAttention", () => {
  it("maps multiple query heads onto one key/value head for GQA", () => {
    const result = qwen2SelfAttentionWithDebug(
      new Float32Array([3, 5, 7, 11]),
      1,
      gqaConfig(),
      attentionWeights({
        query: zeroMatrix(4, 4),
        key: zeroMatrix(2, 4),
        value: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]),
        output: identityMatrix(4),
      }),
    );

    expect(Array.from(result.debug.v)).toEqual([3, 5]);
    expect(Array.from(result.debug.concatenated)).toEqual([3, 5, 3, 5]);
    expect(Array.from(result.output)).toEqual([3, 5, 3, 5]);
  });

  it("matches full-sequence attention when decoding incrementally through the KV cache", () => {
    const config = gqaConfig();
    const input = new Float32Array([
      1, 0, 0, 1,
      0, 1, 1, 0,
      1, 1, 0.5, -0.5,
    ]);
    const weights = attentionWeights({
      query: identityMatrix(4),
      key: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]),
      value: new Float32Array([0, 0, 1, 0, 0, 0, 0, 1]),
      output: identityMatrix(4),
    });

    const full = qwen2SelfAttentionWithDebug(input, 3, config, weights).output;
    const cache: Qwen2LayerKvCache = {
      keys: new Float32Array(3 * config.keyValueHiddenSize),
      values: new Float32Array(3 * config.keyValueHiddenSize),
      length: 0,
    };
    const incremental = new Float32Array(full.length);

    for (let position = 0; position < 3; position += 1) {
      incremental.set(
        qwen2SelfAttentionIncremental(
          input.subarray(position * config.hiddenSize, (position + 1) * config.hiddenSize),
          position,
          config,
          weights,
          cache,
        ),
        position * config.hiddenSize,
      );
    }

    expectArraysClose(incremental, full);
    expect(cache.length).toBe(3);
  });

  it("rejects query head counts that cannot be grouped over KV heads", () => {
    expect(() =>
      qwen2SelfAttentionWithDebug(
        new Float32Array(12),
        1,
        {
          hiddenSize: 12,
          numberOfHeads: 3,
          numberOfKeyValueHeads: 2,
          headDimension: 4,
          keyValueHiddenSize: 8,
          ropeTheta: 10000,
        },
        attentionWeights({
          query: zeroMatrix(12, 12),
          key: zeroMatrix(8, 12),
          value: zeroMatrix(8, 12),
          output: zeroMatrix(12, 12),
        }),
      ),
    ).toThrow("Qwen2 numberOfHeads must be divisible by numberOfKeyValueHeads");
  });
});

function gqaConfig(): Qwen2SelfAttentionConfig {
  return {
    hiddenSize: 4,
    numberOfHeads: 2,
    numberOfKeyValueHeads: 1,
    headDimension: 2,
    keyValueHiddenSize: 2,
    ropeTheta: 10000,
  };
}

function attentionWeights(weights: {
  query: Float32Array;
  key: Float32Array;
  value: Float32Array;
  output: Float32Array;
}): Qwen2SelfAttentionWeights {
  const hiddenSize = Math.sqrt(weights.query.length);
  return {
    query: tensor("q_proj.weight", [hiddenSize, hiddenSize], weights.query),
    queryBias: new Float32Array(hiddenSize),
    key: tensor("k_proj.weight", [weights.key.length / hiddenSize, hiddenSize], weights.key),
    keyBias: new Float32Array(weights.key.length / hiddenSize),
    value: tensor("v_proj.weight", [weights.value.length / hiddenSize, hiddenSize], weights.value),
    valueBias: new Float32Array(weights.value.length / hiddenSize),
    output: tensor("o_proj.weight", [hiddenSize, hiddenSize], weights.output),
  };
}

function identityMatrix(size: number): Float32Array {
  const data = new Float32Array(size * size);
  for (let index = 0; index < size; index += 1) {
    data[index * size + index] = 1;
  }
  return data;
}

function zeroMatrix(rows: number, columns: number): Float32Array {
  return new Float32Array(rows * columns);
}

function tensor(name: string, shape: number[], data: Float32Array): TensorView {
  return {
    name,
    shape,
    byteOffset: 0,
    byteLength: data.byteLength,
    data,
  };
}

function expectArraysClose(actual: Float32Array, expected: Float32Array): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? 0, 6);
  }
}

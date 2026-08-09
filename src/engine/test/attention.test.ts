import { describe, expect, it } from "vitest";
import {
  type CausalSelfAttentionConfig,
  type CausalSelfAttentionWeights,
  causalSelfAttentionWithDebug,
} from "../src/attention";
import type { TensorView } from "../src/loader";

describe("causalSelfAttention", () => {
  it("prevents future value vectors from affecting previous positions", () => {
    const config = attentionConfig(2, 1);
    const baseWeights = attentionWeights({
      query: zeroMatrix(2),
      key: zeroMatrix(2),
      value: identityMatrix(2),
      output: identityMatrix(2),
    });
    const changedFutureWeights = attentionWeights({
      query: zeroMatrix(2),
      key: zeroMatrix(2),
      value: identityMatrix(2),
      output: identityMatrix(2),
    });
    const input = new Float32Array([1, 10, 2, 20, 100, 1000]);
    const changedFutureInput = new Float32Array([1, 10, 2, 20, -7, -70]);

    const base = causalSelfAttentionWithDebug(input, 3, config, baseWeights);
    const changed = causalSelfAttentionWithDebug(changedFutureInput, 3, config, changedFutureWeights);

    expect(Array.from(base.output.slice(0, 4))).toEqual(Array.from(changed.output.slice(0, 4)));
    expect(Array.from(base.output.slice(0, 2))).toEqual([1, 10]);
    expect(Array.from(base.output.slice(2, 4))).toEqual([1.5, 15]);
    expect(base.output[4]).toBeCloseTo(103 / 3, 5);
    expect(base.output[5]).toBeCloseTo(1030 / 3, 4);
    expect(Number.isNaN(base.debug.probabilities[scoreIndex(0, 0, 1, 3)])).toBe(true);
    expect(Number.isNaN(base.debug.probabilities[scoreIndex(0, 1, 2, 3)])).toBe(true);
  });

  it("normalizes each head/query visible attention distribution independently", () => {
    const config = attentionConfig(4, 2);
    const result = causalSelfAttentionWithDebug(
      new Float32Array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]),
      3,
      config,
      attentionWeights({
        query: identityMatrix(4),
        key: identityMatrix(4),
        value: identityMatrix(4),
        output: identityMatrix(4),
      }),
    );

    for (let head = 0; head < config.numberOfHeads; head += 1) {
      for (let queryPosition = 0; queryPosition < 3; queryPosition += 1) {
        let sum = 0;
        for (let keyPosition = 0; keyPosition <= queryPosition; keyPosition += 1) {
          sum += result.debug.probabilities[scoreIndex(head, queryPosition, keyPosition, 3)] ?? 0;
        }
        expect(sum).toBeCloseTo(1, 6);
      }
    }
  });

  it("computes independent attention patterns for multiple heads and concatenates them", () => {
    const config = attentionConfig(4, 2);
    const result = causalSelfAttentionWithDebug(
      new Float32Array([1, 0, 0, 1, 0, 1, 0, 1]),
      2,
      config,
      attentionWeights({
        query: identityMatrix(4),
        key: identityMatrix(4),
        value: identityMatrix(4),
        output: identityMatrix(4),
      }),
    );

    const head0Token1SelfProbability = result.debug.probabilities[scoreIndex(0, 1, 1, 2)] ?? 0;
    const head1Token1SelfProbability = result.debug.probabilities[scoreIndex(1, 1, 1, 2)] ?? 0;

    expect(head0Token1SelfProbability).toBeCloseTo(0.66976154, 6);
    expect(head1Token1SelfProbability).toBeCloseTo(0.5, 6);
    expect(result.debug.headOutput.length).toBe(8);
    expect(result.debug.concatenated.length).toBe(8);
    expect(Array.from(result.debug.concatenated.slice(4, 8))).toEqual([
      result.debug.headOutput[hiddenIndex(1, 0, 0, 4, 2)],
      result.debug.headOutput[hiddenIndex(1, 0, 1, 4, 2)],
      result.debug.headOutput[hiddenIndex(1, 1, 0, 4, 2)],
      result.debug.headOutput[hiddenIndex(1, 1, 1, 4, 2)],
    ]);
    expect(result.output.length).toBe(8);
  });

  it("returns [sequenceLength, hiddenSize] output", () => {
    const result = causalSelfAttentionWithDebug(
      new Float32Array([1, 2, 3, 4, 5, 6]),
      3,
      attentionConfig(2, 1),
      attentionWeights({
        query: zeroMatrix(2),
        key: zeroMatrix(2),
        value: identityMatrix(2),
        output: identityMatrix(2),
      }),
    );

    expect(result.output.length).toBe(3 * 2);
  });

  it("rejects hidden sizes that do not divide evenly into heads", () => {
    expect(() =>
      causalSelfAttentionWithDebug(
        new Float32Array(5),
        1,
        { hiddenSize: 5, numberOfHeads: 2, headDimension: 2 },
        attentionWeights({
          query: identityMatrix(5),
          key: identityMatrix(5),
          value: identityMatrix(5),
          output: identityMatrix(5),
        }),
      ),
    ).toThrow("hiddenSize 5 must be divisible by numberOfHeads 2");
  });

  it("matches a small independently calculated single-head result", () => {
    const result = causalSelfAttentionWithDebug(
      new Float32Array([1, 0, 0, 1]),
      2,
      attentionConfig(2, 1),
      attentionWeights({
        query: identityMatrix(2),
        key: identityMatrix(2),
        value: identityMatrix(2),
        output: identityMatrix(2),
      }),
    );

    expect(Array.from(result.output.slice(0, 2))).toEqual([1, 0]);
    expect(result.debug.scaledScores[scoreIndex(0, 1, 0, 2)]).toBeCloseTo(0, 6);
    expect(result.debug.scaledScores[scoreIndex(0, 1, 1, 2)]).toBeCloseTo(1 / Math.sqrt(2), 6);
    expect(result.debug.probabilities[scoreIndex(0, 1, 0, 2)]).toBeCloseTo(0.33023845, 6);
    expect(result.debug.probabilities[scoreIndex(0, 1, 1, 2)]).toBeCloseTo(0.66976155, 6);
    expect(result.output[2]).toBeCloseTo(0.33023845, 6);
    expect(result.output[3]).toBeCloseTo(0.66976155, 6);
  });
});

function attentionConfig(hiddenSize: number, numberOfHeads: number): CausalSelfAttentionConfig {
  return {
    hiddenSize,
    numberOfHeads,
    headDimension: hiddenSize / numberOfHeads,
  };
}

function attentionWeights(weights: {
  query: Float32Array;
  key: Float32Array;
  value: Float32Array;
  output: Float32Array;
}): CausalSelfAttentionWeights {
  const hiddenSize = Math.sqrt(weights.query.length);
  return {
    query: tensor("query", [hiddenSize, hiddenSize], weights.query),
    key: tensor("key", [hiddenSize, hiddenSize], weights.key),
    value: tensor("value", [hiddenSize, hiddenSize], weights.value),
    output: tensor("output", [hiddenSize, hiddenSize], weights.output),
  };
}

function identityMatrix(size: number): Float32Array {
  const data = new Float32Array(size * size);
  for (let index = 0; index < size; index += 1) {
    data[index * size + index] = 1;
  }
  return data;
}

function zeroMatrix(size: number): Float32Array {
  return new Float32Array(size * size);
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

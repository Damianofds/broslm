import { describe, expect, it } from "vitest";
import {
  dequantizeQ4_0Row,
  dequantizeQ8_0Row,
  embeddingLookupQwen,
  matrixVectorMultiplyQwen,
  type QuantizedTensorView,
} from "../../src/qwen2/quantizedTensor";

describe("Qwen2 quantized tensor helpers", () => {
  it("dequantizes Q4_0 low and high nibbles in GGML row order", () => {
    const values = Array.from({ length: 32 }, (_, index) => (index % 16) - 8);
    const output = new Float32Array(32);

    dequantizeQ4_0Row(q4Row(values), 0, output, 0, 32);

    expect(Array.from(output)).toEqual(values);
  });

  it("multiplies Q4_0 matrix rows by vectors and applies bias", () => {
    const values = Array.from({ length: 32 }, (_, index) => (index % 16) - 8);
    const output = new Float32Array(1);

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 32], "q4_0", q4Row(values)),
      filled(32, 1),
      output,
      { bias: new Float32Array([2]) },
    );

    expect(output[0]).toBe(-14);
  });

  it("dequantizes Q8_0 rows and supports embedding lookups", () => {
    const firstRow = Array.from({ length: 32 }, (_, index) => index - 16);
    const secondRow = Array.from({ length: 32 }, (_, index) => 16 - index);
    const data = concatRows(q8Row(firstRow, 0.5), q8Row(secondRow, 0.5));
    const output = new Float32Array(32);
    const embedding = quantizedTensor("token_embd.weight", [2, 32], "q8_0", data);

    dequantizeQ8_0Row(data, 0, output, 0, 32);
    expect(Array.from(output)).toEqual(firstRow.map((value) => value * 0.5));

    embeddingLookupQwen(embedding, 1, output);
    expect(Array.from(output)).toEqual(secondRow.map((value) => value * 0.5));
  });
});

function quantizedTensor(
  name: string,
  shape: number[],
  type: "q4_0" | "q8_0",
  data: Uint8Array,
): QuantizedTensorView {
  return {
    name,
    shape,
    byteOffset: 0,
    byteLength: data.byteLength,
    type,
    data,
  };
}

function q4Row(values: readonly number[]): Uint8Array {
  if (values.length !== 32) {
    throw new Error("q4Row requires exactly 32 values");
  }
  const row = new Uint8Array(18);
  row[0] = 0x00;
  row[1] = 0x3c;
  for (let index = 0; index < 16; index += 1) {
    row[2 + index] = ((values[index] ?? 0) + 8) | (((values[16 + index] ?? 0) + 8) << 4);
  }
  return row;
}

function q8Row(values: readonly number[], scale: 0.5 | 1): Uint8Array {
  if (values.length !== 32) {
    throw new Error("q8Row requires exactly 32 values");
  }
  const row = new Uint8Array(34);
  row[0] = 0x00;
  row[1] = scale === 0.5 ? 0x38 : 0x3c;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index] ?? 0;
    row[2 + index] = value < 0 ? value + 256 : value;
  }
  return row;
}

function concatRows(...rows: Uint8Array[]): Uint8Array {
  const totalLength = rows.reduce((total, row) => total + row.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const row of rows) {
    output.set(row, offset);
    offset += row.byteLength;
  }
  return output;
}

function filled(length: number, value: number): Float32Array {
  const data = new Float32Array(length);
  data.fill(value);
  return data;
}

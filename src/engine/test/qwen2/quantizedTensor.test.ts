import { describe, expect, it } from "vitest";
import {
  dequantizeIQ1_SRow,
  dequantizeIQ1_MRow,
  dequantizeIQ4_NLRow,
  dequantizeQ2_KRow,
  dequantizeQ4_0Row,
  dequantizeQ5_0Row,
  dequantizeQ5_1Row,
  dequantizeQ8_0Row,
  embeddingLookupQwenGpu,
  embeddingLookupQwen,
  matrixVectorMultiplyQwen,
  matrixVectorMultiplyQwenGpu,
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

  it("dequantizes IQ4_NL rows and supports CPU matvec and embedding lookups", () => {
    const firstRow = iq4NlRow(0x10);
    const secondRow = iq4NlRow(0x21);
    const data = concatRows(firstRow, secondRow);
    const output = new Float32Array(32);
    const expectedFirst = [
      ...Array.from({ length: 16 }, () => -127),
      ...Array.from({ length: 16 }, () => -104),
    ];
    const expectedSecond = [
      ...Array.from({ length: 16 }, () => -104),
      ...Array.from({ length: 16 }, () => -83),
    ];

    dequantizeIQ4_NLRow(data, 0, output, 0, 32);
    expect(Array.from(output)).toEqual(expectedFirst);

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 32], "iq4_nl", firstRow),
      filled(32, 1),
      new Float32Array(1),
    );

    const embedding = quantizedTensor("token_embd.weight", [2, 32], "iq4_nl", data);
    embeddingLookupQwen(embedding, 1, output);
    expect(Array.from(output)).toEqual(expectedSecond);
  });

  it("dequantizes Q5_1 rows and supports CPU matvec and embedding lookups", () => {
    const firstValues = Array.from({ length: 32 }, (_, index) => index);
    const secondValues = Array.from({ length: 32 }, (_, index) => 31 - index);
    const firstRow = q5_1Row(firstValues, 1, 2);
    const secondRow = q5_1Row(secondValues, 1, -1);
    const data = concatRows(firstRow, secondRow);
    const output = new Float32Array(32);

    dequantizeQ5_1Row(data, 0, output, 0, 32);
    expect(Array.from(output)).toEqual(firstValues.map((value) => value + 2));

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 32], "q5_1", firstRow),
      filled(32, 1),
      output,
      { bias: new Float32Array([5]) },
    );
    expect(output[0]).toBe(5 + firstValues.reduce((total, value) => total + value + 2, 0));

    embeddingLookupQwen(quantizedTensor("token_embd.weight", [2, 32], "q5_1", data), 1, output);
    expect(Array.from(output)).toEqual(secondValues.map((value) => value - 1));
  });

  it("dequantizes Q5_0 rows and supports CPU matvec and embedding lookups", () => {
    const firstValues = Array.from({ length: 32 }, (_, index) => index - 16);
    const secondValues = Array.from({ length: 32 }, (_, index) => 15 - index);
    const firstRow = q5_0Row(firstValues);
    const secondRow = q5_0Row(secondValues);
    const data = concatRows(firstRow, secondRow);
    const output = new Float32Array(32);

    dequantizeQ5_0Row(data, 0, output, 0, 32);
    expect(Array.from(output)).toEqual(firstValues);

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 32], "q5_0", firstRow),
      filled(32, 1),
      output,
      { bias: new Float32Array([5]) },
    );
    expect(output[0]).toBe(5 + firstValues.reduce((total, value) => total + value, 0));

    embeddingLookupQwen(quantizedTensor("token_embd.weight", [2, 32], "q5_0", data), 1, output);
    expect(Array.from(output)).toEqual(secondValues);
  });

  it("dequantizes IQ1_S rows and supports CPU matvec and embedding lookups", () => {
    const firstRow = iq1SRow({ signedDelta: false });
    const secondRow = iq1SRow({ signedDelta: true });
    const data = concatRows(firstRow, secondRow);
    const output = new Float32Array(256);

    dequantizeIQ1_SRow(data, 0, output, 0, 256);
    expect(Array.from(output)).toEqual(Array.from({ length: 256 }, () => -0.875));

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 256], "iq1_s", firstRow),
      filled(256, 1),
      output,
      { bias: new Float32Array([4]) },
    );
    expect(output[0]).toBe(-220);

    embeddingLookupQwen(quantizedTensor("token_embd.weight", [2, 256], "iq1_s", data), 1, output);
    expect(Array.from(output)).toEqual(Array.from({ length: 256 }, () => -1.125));
  });

  it("dequantizes IQ1_M rows and supports CPU matvec and embedding lookups", () => {
    const firstRow = iq1MRow({ negativeDelta: false });
    const secondRow = iq1MRow({ negativeDelta: true });
    const data = concatRows(firstRow, secondRow);
    const output = new Float32Array(256);
    const expectedFirstRow = expectedIq1MZeroGrid(firstRow);
    const expectedSecondRow = expectedIq1MZeroGrid(secondRow);

    dequantizeIQ1_MRow(data, 0, output, 0, 256);
    expect(Array.from(output)).toEqual(expectedFirstRow);

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 256], "iq1_m", firstRow),
      filled(256, 1),
      output,
      { bias: new Float32Array([3]) },
    );
    expect(output[0]).toBe(3 + expectedFirstRow.reduce((total, value) => total + value, 0));

    embeddingLookupQwen(quantizedTensor("token_embd.weight", [2, 256], "iq1_m", data), 1, output);
    expect(Array.from(output)).toEqual(expectedSecondRow);
  });

  it("handles padded partial final IQ1_M row blocks", () => {
    const row = concatRows(iq1MRow(), iq1MRow(), iq1MRow(), iq1MRow());
    const output = new Float32Array(896);
    const expected = expectedIq1MZeroGrid(row)
      .concat(expectedIq1MZeroGrid(row.subarray(56)))
      .concat(expectedIq1MZeroGrid(row.subarray(112)))
      .concat(expectedIq1MZeroGrid(row.subarray(168)))
      .slice(0, 896);

    dequantizeIQ1_MRow(row, 0, output, 0, 896);
    expect(Array.from(output)).toEqual(expected);

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 896], "iq1_m", row),
      filled(896, 1),
      new Float32Array(1),
    );
  });

  it("dequantizes Q2_K rows and supports CPU matvec and embedding lookups", () => {
    const firstRow = q2KRow({ scaleNibble: 1, packedQuant: 0xff });
    const secondRow = q2KRow({ scaleNibble: 2, packedQuant: 0x55 });
    const data = concatRows(firstRow, secondRow);
    const output = new Float32Array(256);

    dequantizeQ2_KRow(data, 0, output, 0, 256);
    expect(Array.from(output)).toEqual(Array.from({ length: 256 }, () => 3));

    matrixVectorMultiplyQwen(
      quantizedTensor("weight", [1, 256], "q2_k", firstRow),
      filled(256, 1),
      output,
      { bias: new Float32Array([7]) },
    );
    expect(output[0]).toBe(775);

    embeddingLookupQwen(quantizedTensor("token_embd.weight", [2, 256], "q2_k", data), 1, output);
    expect(Array.from(output)).toEqual(Array.from({ length: 256 }, () => 2));
  });

  it("applies Q2_K group minimums", () => {
    const output = new Float32Array(256);

    dequantizeQ2_KRow(q2KRow({ scaleNibble: 1, minNibble: 2, packedQuant: 0x00 }), 0, output, 0, 256);

    expect(Array.from(output)).toEqual(Array.from({ length: 256 }, () => -2));
  });

  it("rejects CPU-only quantized tensors on WebGPU helper paths", async () => {
    const runtime = {} as Parameters<typeof matrixVectorMultiplyQwenGpu>[0];

    for (const tensor of [
      quantizedTensor("iq1.weight", [1, 256], "iq1_s", iq1SRow()),
      quantizedTensor("iqm.weight", [1, 256], "iq1_m", iq1MRow()),
      quantizedTensor("q2.weight", [1, 256], "q2_k", q2KRow()),
      quantizedTensor("q50.weight", [1, 32], "q5_0", q5_0Row(Array.from({ length: 32 }, () => 0))),
      quantizedTensor("q5.weight", [1, 32], "q5_1", q5_1Row(Array.from({ length: 32 }, () => 0))),
      quantizedTensor("iq4.weight", [1, 32], "iq4_nl", iq4NlRow()),
    ]) {
      await expect(matrixVectorMultiplyQwenGpu(runtime, tensor, filled(256, 1))).rejects.toThrow(
        "only supported on the CPU backend",
      );
      await expect(embeddingLookupQwenGpu(runtime, tensor, 0)).rejects.toThrow(
        "only supported on the CPU backend",
      );
    }
  });
});

function quantizedTensor(
  name: string,
  shape: number[],
  type: "q4_0" | "q5_0" | "q5_1" | "q8_0" | "q2_k" | "iq1_s" | "iq4_nl" | "iq1_m",
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

function iq4NlRow(packed = 0x00): Uint8Array {
  const row = new Uint8Array(18);
  row[0] = 0x00;
  row[1] = 0x3c;
  row.fill(packed, 2);
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

function q5_1Row(values: readonly number[], scale = 1, min = 0): Uint8Array {
  if (values.length !== 32) {
    throw new Error("q5_1Row requires exactly 32 values");
  }
  const row = new Uint8Array(24);
  row[0] = 0x00;
  row[1] = scale === 0.5 ? 0x38 : 0x3c;
  if (min === -1) {
    row[2] = 0x00;
    row[3] = 0xbc;
  } else {
    row[2] = 0x00;
    row[3] = min === 2 ? 0x40 : 0x00;
  }
  let highBits = 0;
  for (let index = 0; index < 16; index += 1) {
    const low = values[index] ?? 0;
    const high = values[16 + index] ?? 0;
    row[8 + index] = (low & 0x0f) | ((high & 0x0f) << 4);
    if (low & 0x10) {
      highBits |= 1 << index;
    }
    if (high & 0x10) {
      highBits |= 1 << (index + 16);
    }
  }
  row[4] = highBits & 0xff;
  row[5] = (highBits >> 8) & 0xff;
  row[6] = (highBits >> 16) & 0xff;
  row[7] = (highBits >> 24) & 0xff;
  return row;
}

function q5_0Row(values: readonly number[], scale = 1): Uint8Array {
  if (values.length !== 32) {
    throw new Error("q5_0Row requires exactly 32 values");
  }
  const row = new Uint8Array(22);
  row[0] = 0x00;
  row[1] = scale === 0.5 ? 0x38 : 0x3c;
  let highBits = 0;
  for (let index = 0; index < 16; index += 1) {
    const low = (values[index] ?? 0) + 16;
    const high = (values[16 + index] ?? 0) + 16;
    row[6 + index] = (low & 0x0f) | ((high & 0x0f) << 4);
    if (low & 0x10) {
      highBits |= 1 << index;
    }
    if (high & 0x10) {
      highBits |= 1 << (index + 16);
    }
  }
  row[2] = highBits & 0xff;
  row[3] = (highBits >> 8) & 0xff;
  row[4] = (highBits >> 16) & 0xff;
  row[5] = (highBits >> 24) & 0xff;
  return row;
}

function q2KRow({
  scaleNibble = 1,
  minNibble = 0,
  packedQuant = 0,
}: {
  scaleNibble?: number;
  minNibble?: number;
  packedQuant?: number;
} = {}): Uint8Array {
  const row = new Uint8Array(84);
  row.fill(((minNibble & 0x0f) << 4) | (scaleNibble & 0x0f), 0, 16);
  row.fill(packedQuant, 16, 80);
  row[80] = 0x00;
  row[81] = 0x3c;
  row[82] = 0x00;
  row[83] = 0x3c;
  return row;
}

function iq1SRow({ signedDelta = false }: { signedDelta?: boolean } = {}): Uint8Array {
  const row = new Uint8Array(50);
  row[0] = 0x00;
  row[1] = 0x3c;
  if (signedDelta) {
    for (let subblock = 0; subblock < 8; subblock += 1) {
      row[35 + subblock * 2] = 0x80;
    }
  }
  return row;
}

function iq1MRow({ negativeDelta = false }: { negativeDelta?: boolean } = {}): Uint8Array {
  const row = new Uint8Array(56);
  row.set(iq1MScales(0x3c00, 1), 48);
  if (negativeDelta) {
    for (let subblock = 0; subblock < 8; subblock += 1) {
      row[32 + subblock * 2] = 0x88;
      row[33 + subblock * 2] = 0x88;
    }
  }
  return row;
}

function iq1MScales(scaleHalf: number, scaleBits: number): Uint8Array {
  const scales = new Uint8Array(8);
  for (let wordIndex = 0; wordIndex < 4; wordIndex += 1) {
    const wordLowBits = scaleBits | (scaleBits << 3);
    const scaleNibble = (scaleHalf >> (4 * wordIndex)) & 0x0f;
    const word = wordLowBits | (scaleNibble << 12);
    scales[wordIndex * 2] = word & 0xff;
    scales[wordIndex * 2 + 1] = (word >> 8) & 0xff;
  }
  return scales;
}

function expectedIq1MZeroGrid(row: Uint8Array): number[] {
  const values: number[] = [];
  for (let subblock = 0; subblock < 8; subblock += 1) {
    const scaleWord = readUint16Le(row, 48 + 2 * Math.floor(subblock / 2));
    const shiftBase = 6 * (subblock % 2);
    const firstScale = 2 * ((scaleWord >> shiftBase) & 0x07) + 1;
    const secondScale = 2 * ((scaleWord >> (shiftBase + 3)) & 0x07) + 1;
    const qh0 = row[32 + 2 * subblock] ?? 0;
    const qh1 = row[32 + 2 * subblock + 1] ?? 0;
    values.push(...Array.from({ length: 8 }, () => firstScale * (qh0 & 0x08 ? -1.125 : -0.875)));
    values.push(...Array.from({ length: 8 }, () => firstScale * (qh0 & 0x80 ? -1.125 : -0.875)));
    values.push(...Array.from({ length: 8 }, () => secondScale * (qh1 & 0x08 ? -1.125 : -0.875)));
    values.push(...Array.from({ length: 8 }, () => secondScale * (qh1 & 0x80 ? -1.125 : -0.875)));
  }
  return values;
}

function readUint16Le(data: Uint8Array, offset: number): number {
  return (data[offset] ?? 0) | ((data[offset + 1] ?? 0) << 8);
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

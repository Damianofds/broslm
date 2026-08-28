import { describe, expect, it } from "vitest";
import {
  GGML_TYPE_IQ1_S,
  GGML_TYPE_IQ1_M,
  GGML_TYPE_IQ4_NL,
  GGML_TYPE_Q2_K,
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q5_0,
  GGML_TYPE_Q5_1,
  parseGguf,
} from "../../src/qwen2/gguf";

describe("GGUF parser tensor byte lengths", () => {
  it("maps IQ1_S tensors to 50-byte GGML blocks of 256 values", () => {
    const gguf = parseGguf(oneTensorGguf("iq.weight", [256], GGML_TYPE_IQ1_S, 50));
    const tensor = gguf.tensors.get("iq.weight");

    expect(tensor?.type).toBe(GGML_TYPE_IQ1_S);
    expect(tensor?.byteLength).toBe(50);
    expect(tensor?.byteOffset).toBe(gguf.tensorDataOffset);
  });

  it("maps Q2_K tensors to 84-byte GGML blocks of 256 values", () => {
    const gguf = parseGguf(oneTensorGguf("q2.weight", [256], GGML_TYPE_Q2_K, 84));
    const tensor = gguf.tensors.get("q2.weight");

    expect(tensor?.type).toBe(GGML_TYPE_Q2_K);
    expect(tensor?.byteLength).toBe(84);
    expect(tensor?.byteOffset).toBe(gguf.tensorDataOffset);
  });

  it("maps Q5_1 tensors to 24-byte GGML blocks of 32 values", () => {
    const gguf = parseGguf(oneTensorGguf("q5.weight", [32], GGML_TYPE_Q5_1, 24));
    const tensor = gguf.tensors.get("q5.weight");

    expect(tensor?.type).toBe(GGML_TYPE_Q5_1);
    expect(tensor?.byteLength).toBe(24);
    expect(tensor?.byteOffset).toBe(gguf.tensorDataOffset);
  });

  it("maps Q5_0 tensors to 22-byte GGML blocks of 32 values", () => {
    const gguf = parseGguf(oneTensorGguf("q5_0.weight", [32], GGML_TYPE_Q5_0, 22));
    const tensor = gguf.tensors.get("q5_0.weight");

    expect(tensor?.type).toBe(GGML_TYPE_Q5_0);
    expect(tensor?.byteLength).toBe(22);
    expect(tensor?.byteOffset).toBe(gguf.tensorDataOffset);
  });

  it("pads Q2_K row byte lengths to whole 256-value blocks", () => {
    const gguf = parseGguf(oneTensorGguf("q2.partial", [896, 2], GGML_TYPE_Q2_K, 672));

    expect(gguf.tensors.get("q2.partial")?.byteLength).toBe(672);
  });

  it("pads IQ1_S row byte lengths to whole 256-value blocks", () => {
    const gguf = parseGguf(oneTensorGguf("iq.partial", [896, 2], GGML_TYPE_IQ1_S, 400));

    expect(gguf.tensors.get("iq.partial")?.byteLength).toBe(400);
  });

  it("maps IQ4_NL tensors to 18-byte GGML blocks of 32 values", () => {
    const gguf = parseGguf(oneTensorGguf("iq4.weight", [32], GGML_TYPE_IQ4_NL, 18));
    const tensor = gguf.tensors.get("iq4.weight");

    expect(tensor?.type).toBe(GGML_TYPE_IQ4_NL);
    expect(tensor?.byteLength).toBe(18);
    expect(tensor?.byteOffset).toBe(gguf.tensorDataOffset);
  });

  it("maps IQ1_M tensors to 56-byte GGML blocks of 256 values", () => {
    const gguf = parseGguf(oneTensorGguf("iqm.weight", [256], GGML_TYPE_IQ1_M, 56));
    const tensor = gguf.tensors.get("iqm.weight");

    expect(tensor?.type).toBe(GGML_TYPE_IQ1_M);
    expect(tensor?.byteLength).toBe(56);
    expect(tensor?.byteOffset).toBe(gguf.tensorDataOffset);
  });

  it("pads IQ1_M row byte lengths to whole 256-value blocks", () => {
    const gguf = parseGguf(oneTensorGguf("iqm.partial", [896, 2], GGML_TYPE_IQ1_M, 448));

    expect(gguf.tensors.get("iqm.partial")?.byteLength).toBe(448);
  });

  it("keeps existing Q4_0 byte-size validation intact", () => {
    const gguf = parseGguf(oneTensorGguf("q4.weight", [32], GGML_TYPE_Q4_0, 18));

    expect(gguf.tensors.get("q4.weight")?.byteLength).toBe(18);
  });
});

function oneTensorGguf(
  name: string,
  dimensions: readonly number[],
  type: number,
  tensorBytes: number,
): ArrayBuffer {
  const writer = new BinaryWriter();
  writer.ascii("GGUF");
  writer.uint32(3);
  writer.uint64(1);
  writer.uint64(0);
  writer.string(name);
  writer.uint32(dimensions.length);
  for (const dimension of dimensions) {
    writer.uint64(dimension);
  }
  writer.uint32(type);
  writer.uint64(0);
  writer.align(32);
  writer.bytes(new Uint8Array(tensorBytes));
  return writer.buffer();
}

class BinaryWriter {
  private readonly bytesList: number[] = [];

  ascii(value: string): void {
    for (const char of value) {
      this.bytesList.push(char.charCodeAt(0));
    }
  }

  string(value: string): void {
    const encoded = new TextEncoder().encode(value);
    this.uint64(encoded.byteLength);
    this.bytes(encoded);
  }

  uint32(value: number): void {
    this.bytesList.push(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff);
  }

  uint64(value: number): void {
    this.uint32(value);
    this.uint32(0);
  }

  bytes(value: Uint8Array): void {
    for (const byte of value) {
      this.bytesList.push(byte);
    }
  }

  align(alignment: number): void {
    while (this.bytesList.length % alignment !== 0) {
      this.bytesList.push(0);
    }
  }

  buffer(): ArrayBuffer {
    return new Uint8Array(this.bytesList).buffer;
  }
}

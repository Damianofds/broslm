import { describe, expect, it } from "vitest";
import {
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q8_0,
  parseGguf,
} from "../../src/qwen2/gguf";

describe("Qwen2.5 Q4_0 GGUF tensors", () => {
  it("calculates Q4_0 tensor storage", () => {
    const gguf = parseGguf(oneTensorGguf("q4.weight", [32], GGML_TYPE_Q4_0, 18));
    expect(gguf.tensors.get("q4.weight")?.byteLength).toBe(18);
  });

  it("calculates Q8_0 tensor storage", () => {
    const gguf = parseGguf(oneTensorGguf("q8.weight", [32], GGML_TYPE_Q8_0, 34));
    expect(gguf.tensors.get("q8.weight")?.byteLength).toBe(34);
  });

  it("rejects tensor formats outside the focused model profile", () => {
    expect(() => parseGguf(oneTensorGguf("unsupported.weight", [32], 19, 8))).toThrow(
      "Unsupported GGUF tensor type 19",
    );
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
  private readonly values: number[] = [];

  ascii(value: string): void {
    for (const character of value) this.values.push(character.charCodeAt(0));
  }

  string(value: string): void {
    const encoded = new TextEncoder().encode(value);
    this.uint64(encoded.byteLength);
    this.bytes(encoded);
  }

  uint32(value: number): void {
    this.values.push(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff);
  }

  uint64(value: number): void {
    this.uint32(value);
    this.uint32(0);
  }

  bytes(value: Uint8Array): void {
    this.values.push(...value);
  }

  align(alignment: number): void {
    while (this.values.length % alignment !== 0) this.values.push(0);
  }

  buffer(): ArrayBuffer {
    return new Uint8Array(this.values).buffer;
  }
}

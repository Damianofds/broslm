export type GgufMetadataValue = number | string | boolean | GgufMetadataValue[];

export interface GgufTensorInfo {
  name: string;
  dimensions: readonly number[];
  type: number;
  offset: number;
  byteOffset: number;
  byteLength: number;
}

export interface GgufFile {
  version: number;
  metadata: ReadonlyMap<string, GgufMetadataValue>;
  tensors: ReadonlyMap<string, GgufTensorInfo>;
  tensorDataOffset: number;
  alignment: number;
}

export const GGUF_TYPE_UINT8 = 0;
export const GGUF_TYPE_INT8 = 1;
export const GGUF_TYPE_UINT16 = 2;
export const GGUF_TYPE_INT16 = 3;
export const GGUF_TYPE_UINT32 = 4;
export const GGUF_TYPE_INT32 = 5;
export const GGUF_TYPE_FLOAT32 = 6;
export const GGUF_TYPE_BOOL = 7;
export const GGUF_TYPE_STRING = 8;
export const GGUF_TYPE_ARRAY = 9;
export const GGUF_TYPE_UINT64 = 10;
export const GGUF_TYPE_INT64 = 11;
export const GGUF_TYPE_FLOAT64 = 12;

export const GGML_TYPE_F32 = 0;
export const GGML_TYPE_F16 = 1;
export const GGML_TYPE_Q4_0 = 2;
export const GGML_TYPE_Q8_0 = 8;

const DEFAULT_ALIGNMENT = 32;

export function parseGguf(buffer: ArrayBuffer): GgufFile {
  const reader = new GgufReader(buffer);
  const magic = reader.readAscii(4);
  if (magic !== "GGUF") {
    throw new Error(`Invalid GGUF magic: ${magic}`);
  }

  const version = reader.readUint32();
  if (version !== 3) {
    throw new Error(`Unsupported GGUF version: ${version}`);
  }

  const tensorCount = reader.readUint64();
  const metadataCount = reader.readUint64();
  const metadata = new Map<string, GgufMetadataValue>();
  for (let index = 0; index < metadataCount; index += 1) {
    const key = reader.readString();
    const valueType = reader.readUint32();
    metadata.set(key, reader.readValue(valueType));
  }

  const rawTensors: Array<Omit<GgufTensorInfo, "byteOffset" | "byteLength">> = [];
  for (let index = 0; index < tensorCount; index += 1) {
    const name = reader.readString();
    const dimensionCount = reader.readUint32();
    const dimensions: number[] = [];
    for (let dimension = 0; dimension < dimensionCount; dimension += 1) {
      dimensions.push(reader.readUint64());
    }
    rawTensors.push({
      name,
      dimensions,
      type: reader.readUint32(),
      offset: reader.readUint64(),
    });
  }

  const alignment = readMetadataNumber(metadata, "general.alignment") ?? DEFAULT_ALIGNMENT;
  if (!Number.isInteger(alignment) || alignment <= 0) {
    throw new Error(`Invalid GGUF alignment: ${alignment}`);
  }
  const tensorDataOffset = alignOffset(reader.offset, alignment);
  const tensors = new Map<string, GgufTensorInfo>();
  for (const tensor of rawTensors) {
    const byteLength = calculateTensorByteLength(tensor.name, tensor.dimensions, tensor.type);
    const byteOffset = tensorDataOffset + tensor.offset;
    if (byteOffset + byteLength > buffer.byteLength) {
      throw new Error(`${tensor.name} points past the end of the GGUF buffer`);
    }
    tensors.set(tensor.name, {
      ...tensor,
      byteOffset,
      byteLength,
    });
  }

  return {
    version,
    metadata,
    tensors,
    tensorDataOffset,
    alignment,
  };
}

export function readMetadataNumber(
  metadata: ReadonlyMap<string, GgufMetadataValue>,
  key: string,
): number | null {
  const value = metadata.get(key);
  if (typeof value !== "number") {
    return null;
  }
  return value;
}

export function readMetadataString(
  metadata: ReadonlyMap<string, GgufMetadataValue>,
  key: string,
): string | null {
  const value = metadata.get(key);
  if (typeof value !== "string") {
    return null;
  }
  return value;
}

export function readMetadataBoolean(
  metadata: ReadonlyMap<string, GgufMetadataValue>,
  key: string,
): boolean | null {
  const value = metadata.get(key);
  if (typeof value !== "boolean") {
    return null;
  }
  return value;
}

export function readMetadataArray<T extends number | string | boolean>(
  metadata: ReadonlyMap<string, GgufMetadataValue>,
  key: string,
  kind: "number" | "string" | "boolean",
): T[] | null {
  const value = metadata.get(key);
  if (!Array.isArray(value)) {
    return null;
  }
  if (value.some((item) => typeof item !== kind)) {
    throw new Error(`GGUF metadata ${key} must be an array of ${kind}`);
  }
  return value as T[];
}

function calculateTensorByteLength(
  name: string,
  dimensions: readonly number[],
  type: number,
): number {
  const elements = product(dimensions);
  switch (type) {
    case GGML_TYPE_F32:
      return elements * 4;
    case GGML_TYPE_F16:
      return elements * 2;
    case GGML_TYPE_Q4_0:
      return quantizedByteLength(name, elements, 32, 18);
    case GGML_TYPE_Q8_0:
      return quantizedByteLength(name, elements, 32, 34);
    default:
      throw new Error(`Unsupported GGUF tensor type ${type} for ${name}`);
  }
}

function quantizedByteLength(
  name: string,
  elements: number,
  blockSize: number,
  typeSize: number,
): number {
  if (elements % blockSize !== 0) {
    throw new Error(`${name} has ${elements} elements, not divisible by quant block ${blockSize}`);
  }
  return (elements / blockSize) * typeSize;
}

function product(values: readonly number[]): number {
  let result = 1;
  for (const value of values) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`Invalid GGUF tensor dimension: ${value}`);
    }
    result *= value;
  }
  return result;
}

function alignOffset(offset: number, alignment: number): number {
  return Math.ceil(offset / alignment) * alignment;
}

class GgufReader {
  readonly view: DataView;
  offset = 0;

  constructor(readonly buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  readAscii(length: number): string {
    this.require(length);
    let value = "";
    for (let index = 0; index < length; index += 1) {
      value += String.fromCharCode(this.view.getUint8(this.offset + index));
    }
    this.offset += length;
    return value;
  }

  readUint8(): number {
    this.require(1);
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  readInt8(): number {
    this.require(1);
    const value = this.view.getInt8(this.offset);
    this.offset += 1;
    return value;
  }

  readUint16(): number {
    this.require(2);
    const value = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return value;
  }

  readInt16(): number {
    this.require(2);
    const value = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return value;
  }

  readUint32(): number {
    this.require(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readInt32(): number {
    this.require(4);
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readFloat32(): number {
    this.require(4);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readFloat64(): number {
    this.require(8);
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  readUint64(): number {
    this.require(8);
    const value = Number(this.view.getBigUint64(this.offset, true));
    this.offset += 8;
    if (!Number.isSafeInteger(value)) {
      throw new Error(`GGUF uint64 exceeds safe integer range: ${value}`);
    }
    return value;
  }

  readInt64(): number {
    this.require(8);
    const value = Number(this.view.getBigInt64(this.offset, true));
    this.offset += 8;
    if (!Number.isSafeInteger(value)) {
      throw new Error(`GGUF int64 exceeds safe integer range: ${value}`);
    }
    return value;
  }

  readString(): string {
    const length = this.readUint64();
    this.require(length);
    const bytes = new Uint8Array(this.buffer, this.offset, length);
    this.offset += length;
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }

  readValue(type: number): GgufMetadataValue {
    switch (type) {
      case GGUF_TYPE_UINT8:
        return this.readUint8();
      case GGUF_TYPE_INT8:
        return this.readInt8();
      case GGUF_TYPE_UINT16:
        return this.readUint16();
      case GGUF_TYPE_INT16:
        return this.readInt16();
      case GGUF_TYPE_UINT32:
        return this.readUint32();
      case GGUF_TYPE_INT32:
        return this.readInt32();
      case GGUF_TYPE_FLOAT32:
        return this.readFloat32();
      case GGUF_TYPE_BOOL:
        return this.readUint8() !== 0;
      case GGUF_TYPE_STRING:
        return this.readString();
      case GGUF_TYPE_UINT64:
        return this.readUint64();
      case GGUF_TYPE_INT64:
        return this.readInt64();
      case GGUF_TYPE_FLOAT64:
        return this.readFloat64();
      case GGUF_TYPE_ARRAY:
        return this.readArray();
      default:
        throw new Error(`Unsupported GGUF metadata type: ${type}`);
    }
  }

  readArray(): GgufMetadataValue[] {
    const itemType = this.readUint32();
    if (itemType === GGUF_TYPE_ARRAY) {
      throw new Error("Nested GGUF metadata arrays are not supported");
    }
    const length = this.readUint64();
    const values: GgufMetadataValue[] = [];
    for (let index = 0; index < length; index += 1) {
      values.push(this.readValue(itemType));
    }
    return values;
  }

  require(length: number): void {
    if (this.offset + length > this.buffer.byteLength) {
      throw new Error("Unexpected end of GGUF buffer");
    }
  }
}

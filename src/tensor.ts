export interface TensorDescriptor {
  shape: number[];
  byteOffset: number;
  byteLength: number;
}

export interface WeightsIndex {
  dtype: "float32";
  byteOrder: "little-endian";
  totalByteLength: number;
  tensorCount: number;
  tensors: Record<string, TensorDescriptor>;
}

export interface TensorView {
  name: string;
  shape: readonly number[];
  byteOffset: number;
  byteLength: number;
  data: Float32Array;
}

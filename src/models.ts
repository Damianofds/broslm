export type ModelId = "qwen";
export type InferenceBackend = "webgpu";

export interface ModelDescriptor {
  id: ModelId;
  label: string;
  shortLabel: string;
  architecture: "qwen2";
  quantization: "Q4_0";
  backend: InferenceBackend;
  source: string;
  minimumStorageBufferBindingSize?: number;
}

export const defaultModelId: ModelId = "qwen";

export const modelCatalog: Readonly<Record<ModelId, ModelDescriptor>> = Object.freeze({
  qwen: Object.freeze({
    id: "qwen",
    label: "Qwen2.5 0.5B Q4_0",
    shortLabel: "Qwen Q4_0",
    architecture: "qwen2",
    quantization: "Q4_0",
    backend: "webgpu",
    source:
      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
    minimumStorageBufferBindingSize: 144_643_072,
  }),
});

export const modelOptions: readonly ModelDescriptor[] = Object.freeze([modelCatalog.qwen]);

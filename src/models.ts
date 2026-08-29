export type ModelId = "qwen" | "qwen_cpu_small";
export type InferenceBackend = "cpu" | "webgpu";

export interface ModelDescriptor {
  id: ModelId;
  label: string;
  shortLabel: string;
  architecture: "qwen2";
  quantization: "Q4_0" | "IQ1_S";
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
  qwen_cpu_small: Object.freeze({
    id: "qwen_cpu_small",
    label: "Qwen2.5 0.5B IQ1_S CPU",
    shortLabel: "Qwen CPU",
    architecture: "qwen2",
    quantization: "IQ1_S",
    backend: "cpu",
    source:
      "https://huggingface.co/legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF/resolve/main/Qwen2.5-0.5B-Instruct.IQ1_S.gguf",
  }),
});

export const modelOptions: readonly ModelDescriptor[] = Object.freeze([
  modelCatalog.qwen,
  modelCatalog.qwen_cpu_small,
]);

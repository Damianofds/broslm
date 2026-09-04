import { describe, expect, it, vi } from "vitest";
import type { Qwen2ModelKvCache } from "../../src/qwen2/attentionCache";
import { preloadQwen2ModelGpu } from "../../src/qwen2/gpuModel";
import type { LoadedQwen2Model, TensorView } from "../../src/qwen2/loader";
import type {
  QuantizedTensorView,
  QwenTensorView,
} from "../../src/qwen2/quantizedTensor";
import type { WebGpuRuntime } from "../../src/runtime/webgpu";

describe("Qwen GPU preload", () => {
  it("uploads and compiles once without changing the logical KV cache", async () => {
    const createBuffer = vi.fn(() => ({ destroy: vi.fn() }));
    const writeBuffer = vi.fn();
    const createComputePipelineAsync = vi.fn(async () => ({}));
    const onSubmittedWorkDone = vi.fn(async () => undefined);
    const runtime = {
      backend: "webgpu",
      adapter: {},
      device: {
        createBuffer,
        createShaderModule: vi.fn(() => ({})),
        createComputePipelineAsync,
        queue: { writeBuffer, onSubmittedWorkDone },
      },
      staticBufferCache: new WeakMap(),
      computePipelineCache: new Map(),
      bindGroupCache: new Map(),
      resourceIds: new WeakMap(),
      nextResourceId: 1,
      shaderF16: false,
    } as unknown as WebGpuRuntime;
    const quantized = {
      name: "quantized",
      shape: [32, 32],
      byteOffset: 0,
      byteLength: 576,
      type: "q4_0",
      data: new Uint8Array(576),
    } satisfies QuantizedTensorView;
    const f32 = {
      name: "f32",
      shape: [32],
      byteOffset: 0,
      byteLength: 128,
      data: new Float32Array(32),
    } satisfies TensorView;
    const model = tinyModel(quantized, f32);
    const cache: Qwen2ModelKvCache = {
      inputIds: [],
      layers: [{ length: 0 }],
      maximumSequenceLength: 2,
      keyValueHiddenSize: 32,
    };

    await preloadQwen2ModelGpu(model, cache, runtime);
    const buffersAfterFirstCall = createBuffer.mock.calls.length;
    const pipelinesAfterFirstCall = createComputePipelineAsync.mock.calls.length;
    await preloadQwen2ModelGpu(model, cache, runtime);

    expect(buffersAfterFirstCall).toBeGreaterThan(0);
    expect(pipelinesAfterFirstCall).toBeGreaterThan(0);
    expect(createBuffer).toHaveBeenCalledTimes(buffersAfterFirstCall);
    expect(createComputePipelineAsync).toHaveBeenCalledTimes(pipelinesAfterFirstCall);
    expect(writeBuffer).toHaveBeenCalled();
    expect(onSubmittedWorkDone).toHaveBeenCalledTimes(2);
    expect(cache.inputIds).toEqual([]);
    expect(cache.layers).toEqual([{ length: 0 }]);
  });
});

function tinyModel(
  quantized: QuantizedTensorView,
  f32: TensorView,
): LoadedQwen2Model {
  return {
    config: {
      architecture: "qwen2",
      vocabularySize: 32,
      hiddenSize: 32,
      intermediateSize: 32,
      numberOfLayers: 1,
      numberOfHeads: 1,
      numberOfKeyValueHeads: 1,
      headDimension: 32,
      keyValueHiddenSize: 32,
      maximumSequenceLength: 2,
      rmsNormEpsilon: 1e-6,
      ropeTheta: 10_000,
      activation: "silu",
      tiedWordEmbeddings: true,
      bosTokenId: 0,
      eosTokenId: 1,
      padTokenId: null,
    },
    gguf: {
      version: 3,
      metadata: new Map(),
      tensors: new Map(),
      tensorDataOffset: 0,
      alignment: 32,
    },
    weightsBuffer: new ArrayBuffer(576),
    tensors: new Map<string, QwenTensorView>([
      [quantized.name, quantized],
      [f32.name, f32],
    ]),
    weights: {
      tokenEmbedding: quantized,
      layers: [{
        index: 0,
        inputLayerNorm: { weight: f32 },
        attention: {
          qProjWeight: quantized,
          qProjBias: f32,
          kProjWeight: quantized,
          kProjBias: f32,
          vProjWeight: quantized,
          vProjBias: f32,
          outProjWeight: quantized,
        },
        postAttentionLayerNorm: { weight: f32 },
        mlp: {
          gateProjWeight: quantized,
          upProjWeight: quantized,
          downProjWeight: quantized,
        },
      }],
      finalNorm: { weight: f32 },
      lmHead: quantized,
    },
  };
}

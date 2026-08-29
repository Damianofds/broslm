import { createBroslmClient, type Broslm, type BroslmOptions } from "./client";
import type { BroslmEnvironment } from "./environment";
import type { WebGpuNavigator } from "./runtime/webgpu";

export * from "./public";

export function createBroslm(options: BroslmOptions = {}): Broslm {
  return createBroslmClient(createNodeEnvironment(), options);
}

function createNodeEnvironment(): BroslmEnvironment {
  let webGpuTarget: WebGpuNavigator | undefined;
  let dawnGpu: unknown;

  return {
    fetchImpl: (input, init) => globalThis.fetch(input, init),
    getWebGpuTarget: async () => {
      if (webGpuTarget) {
        return webGpuTarget;
      }

      let dawn: typeof import("webgpu");
      try {
        dawn = await import("webgpu");
      } catch (error: unknown) {
        const missingPackage =
          error instanceof Error && "code" in error && error.code === "ERR_MODULE_NOT_FOUND";
        const message = missingPackage
          ? "Node WebGPU requires the optional `webgpu` package. Install it alongside `broslm`."
          : `Node WebGPU could not initialize the Dawn binding: ${errorMessage(error)}`;
        throw new Error(message, { cause: error });
      }

      dawnGpu = dawn.create([]);
      webGpuTarget = { gpu: dawnGpu as GPU };
      return webGpuTarget;
    },
    release: () => {
      webGpuTarget = undefined;
      dawnGpu = undefined;
    },
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

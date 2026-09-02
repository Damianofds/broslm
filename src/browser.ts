import { createModelCacheFetch } from "./acquisition/browserCache";
import { createBroslmClient, type Broslm, type BroslmOptions } from "./client";
import type { BroslmEnvironment } from "./environment";

export * from "./public";

export function createBroslm(options: BroslmOptions = {}): Broslm {
  return createBroslmClient(createBrowserEnvironment(), options);
}

function createBrowserEnvironment(): BroslmEnvironment {
  return {
    fetchImpl: createModelCacheFetch(),
    getWebGpuTarget: async () =>
      typeof navigator === "undefined" || !("gpu" in navigator) ? undefined : navigator,
  };
}

import type { WebGpuNavigator } from "./runtime/webgpu";

export interface BroslmEnvironment {
  fetchImpl: typeof fetch;
  getWebGpuTarget(): Promise<WebGpuNavigator | undefined>;
}

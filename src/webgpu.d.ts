declare module "webgpu" {
  export function create(options?: readonly string[]): GPU;
  export const globals: Record<string, unknown>;
}

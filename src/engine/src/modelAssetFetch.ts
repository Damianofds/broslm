export const modelAssetCacheStatusHeader = "x-broslm-model-cache";
export const modelAssetCacheHitHeaderValue = "hit";

export type ModelAssetFetchSource = "cache" | "network";

export function isModelAssetCacheHit(response: Response): boolean {
  return response.headers.get(modelAssetCacheStatusHeader) === modelAssetCacheHitHeaderValue;
}

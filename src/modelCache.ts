import { currentModelExportName } from "./modelExport";
import {
  modelAssetCacheHitHeaderValue,
  modelAssetCacheStatusHeader,
} from "./engine/src/modelAssetFetch";

const cacheName = `broslm-model-${currentModelExportName}`;

export interface ModelCacheFetchOptions {
  fetchImpl?: typeof fetch;
  cacheStorage?: CacheStorage;
}

export function createModelCacheFetch(options: ModelCacheFetchOptions = {}): typeof fetch {
  const fetchImpl = options.fetchImpl ?? fetch;
  const cacheStorage = options.cacheStorage ?? globalThis.caches;

  if (!cacheStorage) {
    return fetchImpl;
  }

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    if (!isCacheableGetRequest(input, init)) {
      return fetchImpl(input, init);
    }

    const request = new Request(input, init);
    const cache = await openModelCache(cacheStorage);
    const cachedResponse = cache ? await matchModelCache(cache, request) : undefined;
    if (cachedResponse) {
      return withCacheHitHeader(cachedResponse);
    }

    const response = await fetchImpl(request);
    if (response.ok && cache) {
      void putModelCache(cache, request, response);
    }
    return response;
  };
}

function withCacheHitHeader(response: Response): Response {
  const headers = new Headers(response.headers);
  headers.set(modelAssetCacheStatusHeader, modelAssetCacheHitHeaderValue);
  return new Response(response.body, {
    headers,
    status: response.status,
    statusText: response.statusText,
  });
}

export async function clearModelCache(
  cacheStorage: CacheStorage | undefined = globalThis.caches,
): Promise<boolean> {
  if (!cacheStorage) {
    return false;
  }
  return cacheStorage.delete(cacheName);
}

async function openModelCache(cacheStorage: CacheStorage): Promise<Cache | null> {
  try {
    return await cacheStorage.open(cacheName);
  } catch {
    return null;
  }
}

async function matchModelCache(cache: Cache, request: Request): Promise<Response | undefined> {
  try {
    return await cache.match(request);
  } catch {
    return undefined;
  }
}

async function putModelCache(cache: Cache, request: Request, response: Response): Promise<void> {
  try {
    await cache.put(request, response.clone());
  } catch {
    // Cache writes can fail because of quota or private-mode storage restrictions.
  }
}

function isCacheableGetRequest(input: RequestInfo | URL, init?: RequestInit): boolean {
  if (init?.method && init.method.toUpperCase() !== "GET") {
    return false;
  }
  if (init?.body) {
    return false;
  }
  if (input instanceof Request && input.method.toUpperCase() !== "GET") {
    return false;
  }
  return true;
}

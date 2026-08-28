import { describe, expect, it, vi } from "vitest";
import {
  modelAssetCacheHitHeaderValue,
  modelAssetCacheStatusHeader,
} from "./engine/src/modelAssetFetch";
import { createModelCacheFetch } from "./modelCache";

describe("createModelCacheFetch", () => {
  it("stores successful GET responses and reuses them", async () => {
    const cacheStorage = new MemoryCacheStorage();
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("cached model bytes"));
    const cachedFetch = createModelCacheFetch({ cacheStorage, fetchImpl });
    const url = "https://example.test/models/current/weights.bin";

    const firstResponse = await cachedFetch(url);
    const secondResponse = await cachedFetch(url);

    await expect(firstResponse.text()).resolves.toBe("cached model bytes");
    await expect(secondResponse.text()).resolves.toBe("cached model bytes");
    expect(secondResponse.headers.get(modelAssetCacheStatusHeader)).toBe(
      modelAssetCacheHitHeaderValue,
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("returns cold network responses before the cache write finishes", async () => {
    let releaseCacheWrite: () => void = () => undefined;
    const cacheWriteGate = new Promise<void>((resolve) => {
      releaseCacheWrite = resolve;
    });
    const cacheStorage = new MemoryCacheStorage({ beforePut: () => cacheWriteGate });
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("streamed model bytes"));
    const cachedFetch = createModelCacheFetch({ cacheStorage, fetchImpl });
    const url = "https://example.test/models/current/weights.bin";

    const response = await cachedFetch(url);

    await expect(response.text()).resolves.toBe("streamed model bytes");
    await expect(cacheStorage.match(url)).resolves.toBeUndefined();

    releaseCacheWrite();
    const cachedResponse = await waitForCachedResponse(cacheStorage, url);
    await expect(cachedResponse.text()).resolves.toBe("streamed model bytes");
  });

  it("does not cache failed responses", async () => {
    const cacheStorage = new MemoryCacheStorage();
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response("missing", { status: 404 }))
      .mockResolvedValueOnce(new Response("available"));
    const cachedFetch = createModelCacheFetch({ cacheStorage, fetchImpl });
    const url = "https://example.test/models/current/config.json";

    const firstResponse = await cachedFetch(url);
    const secondResponse = await cachedFetch(url);

    expect(firstResponse.status).toBe(404);
    await expect(secondResponse.text()).resolves.toBe("available");
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("passes non-GET requests through without touching Cache Storage", async () => {
    const cacheStorage = new MemoryCacheStorage();
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("posted"));
    const cachedFetch = createModelCacheFetch({ cacheStorage, fetchImpl });

    const response = await cachedFetch("https://example.test/models/current/weights.bin", {
      method: "POST",
      body: "ignored",
    });

    await expect(response.text()).resolves.toBe("posted");
    expect(cacheStorage.openCalls).toBe(0);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("still returns the network response when the cache write fails", async () => {
    const cacheStorage = new MemoryCacheStorage({ failWrites: true });
    const fetchImpl = vi.fn<typeof fetch>(async () => new Response("network model bytes"));
    const cachedFetch = createModelCacheFetch({ cacheStorage, fetchImpl });

    const response = await cachedFetch("https://example.test/models/current/weights.bin");

    await expect(response.text()).resolves.toBe("network model bytes");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});

class MemoryCacheStorage implements CacheStorage {
  private readonly cachesByName = new Map<string, MemoryCache>();
  private readonly options: MemoryCacheOptions;
  openCalls = 0;

  constructor(options: MemoryCacheOptions = {}) {
    this.options = options;
  }

  async delete(cacheName: string): Promise<boolean> {
    return this.cachesByName.delete(cacheName);
  }

  async has(cacheName: string): Promise<boolean> {
    return this.cachesByName.has(cacheName);
  }

  async keys(): Promise<string[]> {
    return [...this.cachesByName.keys()];
  }

  async match(request: RequestInfo | URL, options?: MultiCacheQueryOptions): Promise<Response | undefined> {
    for (const cache of this.cachesByName.values()) {
      const response = await cache.match(request, options);
      if (response) {
        return response;
      }
    }
    return undefined;
  }

  async open(cacheName: string): Promise<Cache> {
    this.openCalls += 1;
    const cache = this.cachesByName.get(cacheName) ?? new MemoryCache(this.options);
    this.cachesByName.set(cacheName, cache);
    return cache;
  }
}

interface MemoryCacheOptions {
  failWrites?: boolean;
  beforePut?: () => Promise<void>;
}

class MemoryCache implements Cache {
  private readonly responsesByUrl = new Map<string, Response>();

  constructor(private readonly options: MemoryCacheOptions = {}) {}

  async add(request: RequestInfo | URL): Promise<void> {
    const response = await fetch(request);
    if (!response.ok) {
      throw new TypeError("Failed to fetch");
    }
    await this.put(request, response);
  }

  async addAll(requests: Iterable<RequestInfo | URL>): Promise<void> {
    for (const request of requests) {
      await this.add(request);
    }
  }

  async delete(request: RequestInfo | URL): Promise<boolean> {
    return this.responsesByUrl.delete(requestUrl(request));
  }

  async keys(): Promise<readonly Request[]> {
    return [...this.responsesByUrl.keys()].map((url) => new Request(url));
  }

  async match(request: RequestInfo | URL, _options?: CacheQueryOptions): Promise<Response | undefined> {
    return this.responsesByUrl.get(requestUrl(request))?.clone();
  }

  async matchAll(request?: RequestInfo | URL): Promise<readonly Response[]> {
    if (request) {
      const response = await this.match(request);
      return response ? [response] : [];
    }
    return [...this.responsesByUrl.values()].map((response) => response.clone());
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    if (this.options.failWrites) {
      throw new Error("Cache quota exceeded");
    }
    if (this.options.beforePut) {
      await this.options.beforePut();
    }
    this.responsesByUrl.set(requestUrl(request), response.clone());
  }
}

function requestUrl(request: RequestInfo | URL): string {
  if (request instanceof Request) {
    return request.url;
  }
  return new URL(request).toString();
}

async function waitForCachedResponse(
  cacheStorage: CacheStorage,
  url: string,
): Promise<Response> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const response = await cacheStorage.match(url);
    if (response) {
      return response;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`Timed out waiting for ${url} to be cached`);
}

# broSLM

broSLM runs Qwen2.5 0.5B Instruct Q4_0 entirely in the browser with WebGPU. It owns model download and browser caching, tokenization, GPU inference, sampling, KV-cache lifecycle, and streaming generation.

The browser testbed lives in [Damianofds/broslm-fe-testbed](https://github.com/Damianofds/broslm-fe-testbed).

## Requirements

- A browser with WebGPU support and a compatible GPU adapter.
- `maxStorageBufferBindingSize` of at least 153,151,488 bytes.
- A Web Worker is strongly recommended so inference does not block the UI thread.

broSLM has no CPU fallback and does not support server-side or Node.js inference.

## Install

Install directly from the repository:

```bash
npm install github:Damianofds/broslm
```

## Use

```ts
import { createBroslm } from "broslm";

const broslm = createBroslm();
const support = await broslm.checkModelSupport("qwen");
if (!support.supported) {
  throw new Error(support.reason);
}

await broslm.loadModel("qwen");

for await (const update of broslm.stream("Explain grouped-query attention.", {
  maxTokens: 120,
  temperature: 0.95,
  topK: 10,
})) {
  console.log(update.delta);
}

broslm.dispose();
```

Use `generate` instead of `stream` when only the completed result is needed. Stream chunks expose text deltas through both `delta` and `text`; consumers should append them. Loading and generation accept an `AbortSignal`.

Prompts can also be structured as Qwen ChatML conversations. The final message must be from the user.

```ts
const result = await broslm.generate([
  { role: "system", content: "Answer as a concise TypeScript expert." },
  { role: "user", content: "What is a conditional export?" },
]);

console.log(result.text);
```

## Logging

Library logging defaults to `warn`. Configure it when creating the client:

```ts
const broslm = createBroslm({ logLevel: "info" });
```

Available levels are `error`, `warn`, `info`, and `debug`. INFO includes lifecycle events except names ending in `-progress`; DEBUG includes progress events as well. Every library message starts with `[broslm]`.

## Model

broSLM supports one model profile:

| ID | Model | Backend | Source |
| --- | --- | --- | --- |
| `qwen` | Qwen2.5 0.5B Instruct Q4_0 | WebGPU | Official Qwen GGUF |

The weights are downloaded from Hugging Face and cached with browser CacheStorage when available. They are not included in this npm package and remain subject to their upstream license.

## Development

Node.js is used only for the development toolchain:

```bash
npm install
npm test
npm run typecheck
npm run build
npm run pack:smoke
```

The published package is ESM-only and exposes a single browser runtime.

For GPU performance work, build the package and serve the repository as static files, then open
`benchmark/index.html`. The benchmark reports TTFT, prefill and decode throughput, p50/p95 decode
latency, allocation/readback counters, adapter features and peak tracked GPU-buffer memory through
contexts up to 32,752 cached tokens.

## License

The broSLM source code is available under the MIT License. Downloaded model weights retain their own licenses.

# broSLM

broSLM runs Qwen2-family GGUF models from TypeScript in modern browsers and Node.js. It owns model acquisition, tokenization, CPU/WebGPU inference, sampling, KV-cache lifecycle, and generation observability.

The browser testbed lives in [Damianofds/broslm-fe-testbed](https://github.com/Damianofds/broslm-fe-testbed).

## Install

The package is currently distributed from its versioned Git tag:

```bash
npm install github:Damianofds/broslm#v0.1.0
```

Node GPU inference also needs the optional Dawn binding:

```bash
npm install webgpu
```

CPU inference and browser applications do not install or bundle `webgpu`.

## Use

Browser and Node applications use the same import. Browser applications should run generation inside a Web Worker so inference does not block the UI thread.

```ts
import { createBroslm } from "broslm";

const broslm = createBroslm({
  onEvent(event) {
    console.log(event.type, event);
  },
});

const support = await broslm.checkModelSupport("qwen_cpu_small");
if (!support.supported) {
  throw new Error(support.reason);
}

await broslm.loadModel("qwen_cpu_small");

for await (const update of broslm.stream("Explain grouped-query attention.", {
  maxTokens: 120,
  temperature: 0.95,
  topK: 10,
})) {
  console.log(update.text);
}

broslm.dispose();
```

Use `generate` instead of `stream` when only the completed result is needed. Loading and generation accept an `AbortSignal`; `subscribe` adds additional typed event listeners and returns an unsubscribe function.

Prompts can also be structured as Qwen ChatML conversations. A custom system message and prior assistant turns are optional; the final message must be from the user.

```ts
const result = await broslm.generate([
  { role: "system", content: "Answer as a concise TypeScript expert." },
  { role: "user", content: "What are conditional exports?" },
  { role: "assistant", content: "They select package entry points by environment." },
  { role: "user", content: "Show a minimal example." },
], {
  maxTokens: 120,
});

console.log(result.text);
```

A string prompt remains supported and is converted internally to a single user message. Both forms use the same generation, streaming, token-counting, and cancellation APIs.

## Models

| ID | Model | Backend | Source |
| --- | --- | --- | --- |
| `qwen` | Qwen2.5 0.5B Instruct Q4_0 | WebGPU only | Official Qwen GGUF |
| `qwen_cpu_small` | Qwen2.5 0.5B Instruct IQ1_S | CPU only | legraphista IMat GGUF |

Model weights are downloaded from Hugging Face and are not included in this npm package. Each model remains subject to its upstream license.

In browsers, successful model responses are cached with CacheStorage when available. Node uses direct fetch without persistent caching. A future acquisition provider can replace the direct-download implementation without changing the inference API.

## Development

Requires Node.js 18 or newer.

```bash
npm install
npm test
npm run typecheck
npm run build
npm run pack:smoke
```

The package is ESM-only. Conditional exports select the browser or Node runtime while retaining the same public API. `webgpu` is loaded dynamically only when Node requests the GPU model.

## License

The broSLM source code is available under the MIT License. Downloaded model weights retain their own licenses.

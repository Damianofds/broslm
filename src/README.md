# broSLM

This folder contains the TypeScript and React browser app for broSLM. The app loads Qwen2-family GGUF models inside a dedicated inference Web Worker and shows each loading stage in the browser.

Vite runs the dev server, bundles the React page, bundles the worker and shared TypeScript engine code, serves local model files from `../models`, and copies known local GGUF folders into `dist/models/` when present.

## Requirements

- Node.js compatible with Vite 8. Vite currently requires Node.js 20.19+ or 22.12+.
- WebGPU for the default Qwen2.5 0.5B Q4_0 model.
- No WebGPU requirement for the CPU-only Qwen2.5 0.5B IQ1_S model.

## Install

```bash
npm install
```

## Local Development

```bash
npm run dev
```

Open the local URL printed by Vite, usually:

```text
http://localhost:5173/
```

The page is a three-section scroll experience: overview, model loading and inspection, then the textarea chat demo. Select either Qwen model in the second section, click `Start`, then generate from the final section after the model is ready.

## Build

```bash
npm run build
```

The build output is written to `dist/`. Local GGUF folders are copied into `dist/models/` when they exist:

```text
dist/models/qwen2.5-0.5b-instruct-q4_0/
dist/models/qwen2.5-0.5b-instruct-iq1_s/
```

To run TypeScript's project type check separately:

```bash
npm run typecheck
```

## Unit Tests

```bash
npm test
```

The test suite uses Vitest and covers engine primitives, Qwen2 GGUF parsing, quantization, attention, model behavior, sampling, cache behavior, and the browser-facing catalog/cache helpers.

## Preview

```bash
npm run preview
```

## Notes

- The UI thread only starts and observes the load.
- The model is retained inside `src/modelWorker.ts`, which imports the Qwen2 loader and model engine.
- Qwen prompts use a hidden ChatML template for generation while keeping the textarea text clean.
- Vite serves the local `../models` directory at `/models` during development.

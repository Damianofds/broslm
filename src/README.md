# broSLM

This folder contains the TypeScript and React browser app for broSLM. The app loads either a raw TinyStories GPT-Neo export or an optional Qwen2.5 0.5B GGUF model inside a dedicated inference Web Worker and shows each loading stage in the browser.

Vite is the build tool for the whole browser deliverable. It runs the dev server, bundles the React page, bundles the worker and shared TypeScript engine code, and copies the selected raw model export into `dist/models/` for production preview or deployment.

## Requirements

- Node.js compatible with Vite 8. Vite currently requires Node.js 20.19+ or 22.12+.
- The exported model files must exist under `../models/output_20260726_105535/`:
  - `config.json`
  - `weights.json`
  - `weights.bin`
- The browser prompt demo uses `public/tokenizer/tinystories-tokenizer.json`.
- Qwen2 runs download the official Q4_0 GGUF directly from Hugging Face.

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

The page is a three-section scroll experience: overview, model loading and
inspection, then the textarea chat demo. Select TinyStories or Qwen in the
second section, click `Start`, then generate from the final section after the
model is ready.

## Build

```bash
npm run build
```

The build output is written to `dist/`. Vite copies the selected model export to:

```text
dist/models/output_20260726_105535/
```

To run TypeScript's project type check separately:

```bash
npm run typecheck
```

## Unit Tests

```bash
npm test
```

The test suite uses Vitest and covers the TypeScript engine primitives, GPT-Neo
attention/MLP/transformer behavior, Qwen2 GGUF/quantization/attention/model
behavior, and the TinyStories forward-pass smoke test. Tests live under
`engine/test/`.

Run the TinyStories smoke test by itself:

```bash
npm test -- modelForward
```

More explicit Vitest form:

```bash
npm exec vitest -- run engine/test/modelForward.test.ts --reporter verbose
```

Run only the smoke test case by name:

```bash
npm exec vitest -- run -t "predicts the oracle next token"
```

Run the smoke test in watch mode while editing:

```bash
npm exec vitest -- watch engine/test/modelForward.test.ts
```

The smoke-test guard uses the oracle prompt `Once upon a time`, token IDs
`[7454, 2402, 257, 640]`, and expects next token ID `11`, which decodes to `,`.
The test loads the real raw export from `../models/output_20260726_105535/`.

To debug with breakpoints, add `debugger;` in `engine/test/modelForward.test.ts`
or the engine code, then run:

```bash
npm exec vitest -- run engine/test/modelForward.test.ts --inspectBrk --no-file-parallelism --maxWorkers=1
```

Attach a Node debugger to `127.0.0.1:9229`. In Chrome, open:

```text
chrome://inspect
```

For VS Code, this attach configuration works:

```json
{
  "type": "node",
  "request": "attach",
  "name": "Attach Vitest",
  "port": 9229,
  "autoAttachChildProcesses": true,
  "skipFiles": ["<node_internals>/**"]
}
```

## Preview

```bash
npm run preview
```

## Notes

- The UI thread only starts and observes the load.
- The model is retained inside `src/modelWorker.ts`, which imports `engine/src/gpt-neo/loader.ts`.
- The active model export is configured in `src/modelExport.ts`.
- Qwen2 support is available from the second-section model selector. Qwen uses a hidden chat template for generation while keeping the textarea text clean.
- Vite serves the local `../models` directory at `/models` during development and copies the selected export during production builds.

# broSLM

This folder contains the TypeScript and React browser app for broSLM. The app loads a raw TinyStories GPT-Neo export inside a dedicated inference Web Worker and shows each loading stage in the browser.

Vite is the build tool for the whole browser deliverable. It runs the dev server, bundles the React page, bundles the worker and shared TypeScript engine code, and copies the selected raw model export into `dist/models/` for production preview or deployment.

## Requirements

- Node.js compatible with Vite 8. Vite currently requires Node.js 20.19+ or 22.12+.
- The exported model files must exist under `../models/output_20260726_105535/`:
  - `config.json`
  - `weights.json`
  - `weights.bin`

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

The unit test suite uses Vitest and currently covers the pure TypeScript engine primitives under `engine/src/primitives/`. Tests live in the mirrored `engine/test/primitives/` tree.

## Preview

```bash
npm run preview
```

## Notes

- The UI thread only starts and observes the load.
- The model is retained inside `src/modelWorker.ts`, which imports `engine/src/loader.ts`.
- The active model export is configured in `src/modelExport.ts`.
- Vite serves the local `../models` directory at `/models` during development and copies the selected export during production builds.

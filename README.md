# broSLM

broSLM stands for Browser Small Language Model. The project runs Qwen2-family GGUF models directly in the browser with a TypeScript engine, React UI, and dedicated inference Web Worker.

The repo is split into two working areas:

- `src/` contains the TypeScript and React browser app. Vite is the build tool for the browser deliverable. See `src/README.md`.
- `models/` can hold optional local GGUF files served during development or copied into production builds. See `models/README.md`.

## Current Flow

1. Use the TypeScript workspace to run or build the browser app with Vite.
2. Load the default Qwen2.5 0.5B Instruct Q4_0 GGUF from Hugging Face, or place a local copy at `models/qwen2.5-0.5b-instruct-q4_0/model.gguf`.
3. For CPU-only smaller runs, use the Qwen2.5 0.5B Instruct IQ1_S GGUF from `legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF`, or place it at `models/qwen2.5-0.5b-instruct-iq1_s/model.gguf`.

## Build And Run

Install the browser app dependencies from the Vite workspace:

```bash
cd src
npm install
```

Run tests and type checks:

```bash
npm test
npm run typecheck
```

Run the development server:

```bash
npm run dev
```

Build and preview a production build:

```bash
npm run build
npm run preview
```

## Read Next

- TypeScript browser app and Vite commands: `src/README.md`
- GGUF model layout: `models/README.md`

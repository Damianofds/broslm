# broSLM

broSLM stands for Browser Small Language Model. The project explores running small causal language models directly in the browser. The app can load the default Hugging Face TinyStories GPT-Neo raw export or an optional Qwen2.5 0.5B GGUF model from a local model folder.

The repo is split into three working areas:

- `src/` contains the TypeScript and React browser app. Vite is the build tool for the full browser deliverable, including the selected raw model export. See `src/README.md`.
- `models/` contains timestamped raw model exports consumed by the browser app. See `models/README.md`.
- `python-oracle/` contains the Python reference workflow for downloading the Hugging Face model, running it with PyTorch, exporting raw tensors, and validating exports. See `python-oracle/README.md`.

## Current Flow

1. Use the Python workspace to download or validate the TinyStories model and export raw files.
2. Keep completed TinyStories exports in `models/output_<YYYYmmdd_HHMMSS>/`.
3. Keep optional Qwen2 GGUF files in `models/qwen2.5-0.5b-instruct-q4_0/model.gguf`.
4. Use the TypeScript workspace to run or build the browser app with Vite.

## Build And Run

Install the browser app dependencies from the Vite workspace:

```bash
cd src
npm install
```

### Dev

Run the engine unit tests:

```bash
npm test
npm run typecheck
```

Run the development server:

```bash
npm run dev
```

### Prod

Build the browser app:

```bash
npm run build
```

Preview a production build:

```bash
npm run preview
```

## Read Next

- Python reference and export tools: `python-oracle/README.md`
- TypeScript browser app and Vite commands: `src/README.md`
- Raw model export convention: `models/README.md`

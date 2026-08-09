# broSLM

broSLM stands for Browser Small Language Model. The project explores running a small causal language model directly in the browser by exporting a Hugging Face TinyStories GPT-Neo checkpoint into simple raw files and loading those files from a TypeScript inference worker.

The repo is split into three working areas:

- `src/` contains the TypeScript and React browser app. Vite is the build tool for the full browser deliverable, including the selected raw model export. See `src/README.md`.
- `models/` contains timestamped raw model exports consumed by the browser app. See `models/README.md`.
- `python-oracle/` contains the Python reference workflow for downloading the Hugging Face model, running it with PyTorch, exporting raw tensors, and validating exports. See `python-oracle/README.md`.

## Current Flow

1. Use the Python workspace to download or validate the TinyStories model and export raw files.
2. Keep completed exports in `models/output_<YYYYmmdd_HHMMSS>/`.
3. Use the TypeScript workspace to run or build the browser app with Vite.

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

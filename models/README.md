# broSLM Models

This folder stores optional local GGUF files for browser-side Qwen2 inference. If a local file is absent, the app uses the Hugging Face URL from `src/modelCatalog.ts`.

## Local Layout

Optimized WebGPU model:

```text
qwen2.5-0.5b-instruct-q4_0/
+-- model.gguf
```

CPU-only small model:

```text
qwen2.5-0.5b-instruct-iq1_s/
+-- model.gguf
```

## Model Sources

- `qwen2.5-0.5b-instruct-q4_0/model.gguf`: official `Qwen/Qwen2.5-0.5B-Instruct-GGUF` Q4_0 file.
- `qwen2.5-0.5b-instruct-iq1_s/model.gguf`: `legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF` `Qwen2.5-0.5B-Instruct.IQ1_S.gguf` file.

Keep each GGUF folder self-contained so the browser app or engine can load it from `/models/<folder-name>/`.

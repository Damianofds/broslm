# broSLM Models

This folder stores raw model exports and local GGUF files for browser-side inference experiments.

## Current Model

The TypeScript app currently uses:

```text
output_20260726_105535
```

That export contains the raw files expected by the loader:

```text
config.json
weights.json
weights.bin
```

## Qwen2 GGUF Experiments

The Qwen2 engine can load a single GGUF file. A convenient local layout is:

```text
qwen2.5-0.5b-instruct-q4_0/
+-- model.gguf
```

For the official Qwen2.5 0.5B Instruct Q4_0 export, place
`qwen2.5-0.5b-instruct-q4_0.gguf` in that folder as `model.gguf`, or pass the
actual filename as `ggufPath` to `loadQwen2Model`.

## Folder Naming

Export folders use this convention:

```text
output_<YYYYmmdd_HHMMSS>
```

The timestamp records when the Python exporter created the raw export. Keep each export or GGUF folder self-contained so the browser app or engine can load it from `/models/<folder-name>/`.

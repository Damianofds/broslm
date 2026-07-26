# broSLM Models

This folder stores raw model exports for the browser app.

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

## Folder Naming

Export folders use this convention:

```text
output_<YYYYmmdd_HHMMSS>
```

The timestamp records when the Python exporter created the raw export. Keep each export self-contained so the browser app can load it from `/models/<folder-name>/`.

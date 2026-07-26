# TinyStories Oracle

This project runs a small Hugging Face causal language model and prints the most likely next token for the prompt `Once upon a time`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in each new shell before using the scripts.

## Run

```bash
python run_tinystories.py
```

The script downloads `roneneldan/TinyStories-3M` into `artifacts/hf-cache/` and writes the latest result to `artifacts/outputs/tinystories_result.txt`.

## Interact with the model

After running `run_tinystories.py` once, use the local model cache:

```bash
python interact_tinystories.py
```

Then enter prompts one at a time. Press `Ctrl-D` or submit an empty prompt to exit.

The default completion length is 100 new tokens. To change it:

```bash
python interact_tinystories.py --max-new-tokens 40
```

Programmatic use:

```python
from interact_tinystories import generate_text

story = generate_text("Once upon a time", max_new_tokens=40)
print(story)
```

`interact_tinystories.py` stops if the local model files are missing or incomplete and suggests running `run_tinystories.py`.

## Export raw model files

After the model has been downloaded locally, export it to a TypeScript-friendly raw binary format:

```bash
python export_raw_gptneo.py
```

By default, the exporter reads the local `roneneldan/TinyStories-3M` Hugging Face snapshot from `artifacts/hf-cache/` and writes a new timestamped folder:

```text
../models/output_<YYYYmmdd_HHMMSS>/
+-- config.json
+-- weights.json
+-- weights.bin
```

`config.json` contains the model architecture fields needed by a parser, including vocabulary size, hidden size, layer count, head count, head dimension, sequence length, activation, layer norm epsilon, GPT-Neo attention layers, local attention window size, and token IDs.

`weights.json` contains the tensor index:

```json
{
  "dtype": "float32",
  "byteOrder": "little-endian",
  "totalByteLength": 123456,
  "tensorCount": 10,
  "tensors": {
    "transformer.wte.weight": {
      "shape": [50257, 128],
      "byteOffset": 0,
      "byteLength": 25731584
    }
  }
}
```

`weights.bin` contains every tensor from `model.state_dict()` concatenated as contiguous little-endian FP32 values. The exporter logs each step and each tensor as it is written.

To export a specific local model directory:

```bash
python export_raw_gptneo.py --model-path /path/to/huggingface/model
```

To choose the output directory manually:

```bash
python export_raw_gptneo.py --output-dir ../models/output_manual
```

The exporter uses local files only by default. If you want Hugging Face Transformers to download missing files, pass:

```bash
python export_raw_gptneo.py --allow-download
```

## Validate raw tensors

To verify that the raw export is byte-accurate against the original PyTorch tensors:

```bash
python validate_raw_tensors.py
```

By default, the validator reads the cached TinyStories model and compares it against the newest complete `../models/output_*` directory. It checks that:

- `weights.bin` has the expected total byte length
- every tensor listed in `weights.json` exists in the original `state_dict()`
- every tensor shape and byte length matches
- every raw FP32 tensor value matches the original PyTorch tensor exactly

To validate a specific export directory:

```bash
python validate_raw_tensors.py --export-dir ../models/output_YYYYmmdd_HHMMSS
```

To validate against a specific original Hugging Face model directory:

```bash
python validate_raw_tensors.py --model-path /path/to/huggingface/model --export-dir ../models/output_YYYYmmdd_HHMMSS
```

## Validate raw model output

To rebuild a PyTorch model from the raw export and compare its logits against the original Hugging Face model:

```bash
python validate_raw_forward.py
```

By default, this uses the prompt `Once upon a time`, the cached TinyStories model, and the newest complete `../models/output_*` directory. It:

- loads the original model and tokenizer
- reconstructs a fresh PyTorch model from `weights.bin` and `weights.json`
- runs both models on the same prompt
- compares the full logits tensor with `torch.allclose`
- compares the predicted next token

To test a different prompt:

```bash
python validate_raw_forward.py --prompt "The little dragon"
```

To validate a specific export:

```bash
python validate_raw_forward.py --export-dir ../models/output_YYYYmmdd_HHMMSS
```

To validate a specific original model and export:

```bash
python validate_raw_forward.py --model-path /path/to/huggingface/model --export-dir ../models/output_YYYYmmdd_HHMMSS
```

The `artifacts/` directory and virtual environment are ignored by version control.

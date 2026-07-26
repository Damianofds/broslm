import os
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "hf-cache"
OUTPUT_DIR = ARTIFACTS_DIR / "outputs"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_ID = "roneneldan/TinyStories-3M"

ARTIFACTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model.eval()

prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs)

logits = output.logits
next_token = torch.argmax(logits[0, -1]).item()

print("Input IDs:", inputs["input_ids"].tolist())
print("Next token:", next_token)
print("Decoded:", tokenizer.decode([next_token]))

result = (
    f"Input IDs: {inputs['input_ids'].tolist()}\n"
    f"Next token: {next_token}\n"
    f"Decoded: {tokenizer.decode([next_token])}\n"
)
(OUTPUT_DIR / "tinystories_result.txt").write_text(result, encoding="utf-8")

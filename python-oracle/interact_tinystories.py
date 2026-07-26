import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "roneneldan/TinyStories-3M"
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "artifacts" / "hf-cache"
MODEL_CACHE_DIR = CACHE_DIR / "models--roneneldan--TinyStories-3M"
SNAPSHOTS_DIR = MODEL_CACHE_DIR / "snapshots"


def find_local_model_path() -> Path:
    if not SNAPSHOTS_DIR.exists():
        raise SystemExit(
            "TinyStories model is not present locally. "
            "Run `.venv/bin/python run_tinystories.py` first to download it."
        )

    for snapshot in sorted(SNAPSHOTS_DIR.iterdir(), reverse=True):
        if (
            (snapshot / "config.json").exists()
            and (snapshot / "pytorch_model.bin").exists()
            and (snapshot / "tokenizer.json").exists()
        ):
            return snapshot

    raise SystemExit(
        "TinyStories model cache is incomplete. "
        "Run `.venv/bin/python run_tinystories.py` first to download it."
    )


def load_model():
    model_path = find_local_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    print('######### model.config #########')
    print(model.config)
    print('############# model ############')
    print(model)
    return tokenizer, model


def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    do_sample: bool = True,
    tokenizer=None,
    model=None,
) -> str:
    if tokenizer is None or model is None:
        tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        generated = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interact with the locally downloaded TinyStories model."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    tokenizer, model = load_model()
    print("TinyStories interactive prompt. Press Ctrl-D or enter an empty prompt to exit.")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            break

        text = generate_text(
            prompt,
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
            model=model,
        )
        print("\nCompletion:")
        print(text)


if __name__ == "__main__":
    main()

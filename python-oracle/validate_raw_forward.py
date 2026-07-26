import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export_raw_gptneo import CACHE_DIR, find_default_model_path
from validate_raw_tensors import find_latest_export_dir, load_weights_index, validate_file_size


LOGGER = logging.getLogger("validate_raw_forward")

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a PyTorch model from raw exported tensors and compare its "
            "forward-pass logits against the original Hugging Face model."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to the original local Hugging Face model directory. Defaults to "
            "the cached roneneldan/TinyStories-3M snapshot."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing config.json, weights.json, and weights.bin. "
            "Defaults to the newest ../models/output_* directory."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt used for the forward-pass comparison.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for torch.allclose.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for torch.allclose.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download missing original model files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_original_model_and_tokenizer(model_path: Path, local_files_only: bool) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading original tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )

    LOGGER.info("Loading original model from %s", model_path)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    original_model.eval()
    LOGGER.info("Loaded original model class: %s", original_model.__class__.__name__)
    return original_model, tokenizer


def read_exported_state_dict(
    reference_state_dict: dict[str, Any],
    weights_index: dict[str, Any],
    weights_bin_path: Path,
) -> dict[str, torch.Tensor]:
    LOGGER.info("Reading raw weights from %s", weights_bin_path)
    weights_blob = weights_bin_path.read_bytes()

    exported_tensors = weights_index["tensors"]
    missing = sorted(set(reference_state_dict) - set(exported_tensors))
    extra = sorted(set(exported_tensors) - set(reference_state_dict))
    if missing:
        raise ValueError(f"weights.json is missing original tensors: {missing}")
    if extra:
        raise ValueError(f"weights.json contains unknown tensors: {extra}")

    raw_state_dict: dict[str, torch.Tensor] = {}
    LOGGER.info("Reconstructing %s tensors from raw export", len(reference_state_dict))
    for index, (name, reference_tensor) in enumerate(reference_state_dict.items(), start=1):
        tensor_info = exported_tensors[name]
        shape = tensor_info.get("shape")
        byte_offset = tensor_info.get("byteOffset")
        byte_length = tensor_info.get("byteLength")

        if shape != list(reference_tensor.shape):
            raise ValueError(
                f"{name} shape mismatch: exported {shape}, original {list(reference_tensor.shape)}"
            )
        if byte_length != reference_tensor.numel() * 4:
            raise ValueError(
                f"{name} byteLength is {byte_length}, expected {reference_tensor.numel() * 4}"
            )
        if byte_offset + byte_length > len(weights_blob):
            raise ValueError(f"{name} points past the end of weights.bin")

        array = np.frombuffer(
            weights_blob,
            dtype="<f4",
            count=byte_length // 4,
            offset=byte_offset,
        ).reshape(shape)

        raw_state_dict[name] = torch.from_numpy(array.copy()).to(dtype=reference_tensor.dtype)
        LOGGER.info("Reconstructed tensor %s/%s: %s", index, len(reference_state_dict), name)

    return raw_state_dict


def build_raw_model(original_model: Any, raw_state_dict: dict[str, torch.Tensor]) -> Any:
    from transformers import AutoModelForCausalLM

    LOGGER.info("Instantiating fresh model from original config")
    raw_model = AutoModelForCausalLM.from_config(original_model.config)
    load_result = raw_model.load_state_dict(raw_state_dict, strict=True)
    raw_model.eval()
    LOGGER.info("Loaded raw tensors into fresh model")
    LOGGER.debug("load_state_dict result: %s", load_result)
    return raw_model


def compare_forward_outputs(
    original_model: Any,
    raw_model: Any,
    tokenizer: Any,
    prompt: str,
    atol: float,
    rtol: float,
) -> None:
    LOGGER.info("Tokenizing prompt: %r", prompt)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        LOGGER.info("Running original model forward pass")
        original_logits = original_model(**inputs).logits
        LOGGER.info("Running raw-restored model forward pass")
        raw_logits = raw_model(**inputs).logits

    exact_equal = torch.equal(original_logits, raw_logits)
    close = torch.allclose(original_logits, raw_logits, atol=atol, rtol=rtol)
    max_abs_diff = torch.max(torch.abs(original_logits - raw_logits)).item()

    original_next_token_id = torch.argmax(original_logits[0, -1]).item()
    raw_next_token_id = torch.argmax(raw_logits[0, -1]).item()
    original_next_token = tokenizer.decode([original_next_token_id])
    raw_next_token = tokenizer.decode([raw_next_token_id])

    LOGGER.info("Logits shape: %s", list(original_logits.shape))
    LOGGER.info("Exact logits equality: %s", exact_equal)
    LOGGER.info("Allclose logits equality: %s with atol=%s rtol=%s", close, atol, rtol)
    LOGGER.info("Maximum absolute logits difference: %.12g", max_abs_diff)
    LOGGER.info(
        "Original next token: id=%s text=%r",
        original_next_token_id,
        original_next_token,
    )
    LOGGER.info("Raw-restored next token: id=%s text=%r", raw_next_token_id, raw_next_token)

    if not close:
        raise ValueError(
            "Forward-pass logits differ beyond tolerance: "
            f"max_abs_diff={max_abs_diff}, atol={atol}, rtol={rtol}"
        )
    if original_next_token_id != raw_next_token_id:
        raise ValueError(
            "Next-token prediction differs: "
            f"original={original_next_token_id}, raw={raw_next_token_id}"
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    model_path = args.model_path.resolve() if args.model_path else find_default_model_path()
    export_dir = args.export_dir.resolve() if args.export_dir else find_latest_export_dir()
    local_files_only = not args.allow_download

    weights_index = load_weights_index(export_dir)
    weights_bin_path = export_dir / "weights.bin"
    validate_file_size(weights_bin_path, weights_index)

    original_model, tokenizer = load_original_model_and_tokenizer(model_path, local_files_only)
    raw_state_dict = read_exported_state_dict(
        original_model.state_dict(),
        weights_index,
        weights_bin_path,
    )
    raw_model = build_raw_model(original_model, raw_state_dict)
    compare_forward_outputs(
        original_model,
        raw_model,
        tokenizer,
        args.prompt,
        args.atol,
        args.rtol,
    )

    LOGGER.info("Forward validation complete: raw-restored model matches original output")


if __name__ == "__main__":
    main()

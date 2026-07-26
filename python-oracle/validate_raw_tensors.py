import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from export_raw_gptneo import CACHE_DIR, PROJECT_DIR, find_default_model_path


LOGGER = logging.getLogger("validate_raw_tensors")

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exported raw tensor files by comparing weights.bin and "
            "weights.json against the original Hugging Face model state_dict."
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


def find_latest_export_dir() -> Path:
    models_dir = PROJECT_DIR.parent / "models"
    LOGGER.info("Looking for latest export in %s", models_dir)
    candidates = sorted(models_dir.glob("output_*"), reverse=True)
    for candidate in candidates:
        if (
            (candidate / "config.json").exists()
            and (candidate / "weights.json").exists()
            and (candidate / "weights.bin").exists()
        ):
            LOGGER.info("Using latest export directory: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "No complete ../models/output_* export directory found. Pass --export-dir."
    )


def load_model(model_path: Path, local_files_only: bool) -> Any:
    from transformers import AutoModelForCausalLM

    LOGGER.info("Loading original model from %s", model_path)
    LOGGER.info("local_files_only=%s", local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    model.eval()
    LOGGER.info("Loaded model class: %s", model.__class__.__name__)
    return model


def load_weights_index(export_dir: Path) -> dict[str, Any]:
    weights_json_path = export_dir / "weights.json"
    LOGGER.info("Loading tensor index from %s", weights_json_path)
    weights_index = json.loads(weights_json_path.read_text(encoding="utf-8"))

    if weights_index.get("dtype") != "float32":
        raise ValueError(f"Expected dtype float32, got {weights_index.get('dtype')}")
    if weights_index.get("byteOrder") != "little-endian":
        raise ValueError(
            f"Expected byteOrder little-endian, got {weights_index.get('byteOrder')}"
        )
    if not isinstance(weights_index.get("tensors"), dict):
        raise ValueError("weights.json must contain a tensors object")

    return weights_index


def validate_file_size(weights_bin_path: Path, weights_index: dict[str, Any]) -> None:
    actual_size = weights_bin_path.stat().st_size
    expected_size = weights_index.get("totalByteLength")
    LOGGER.info("weights.bin size: %s bytes", actual_size)
    if actual_size != expected_size:
        raise ValueError(
            f"weights.bin is {actual_size} bytes, weights.json says {expected_size}"
        )


def validate_tensor_directory(
    state_dict: dict[str, Any],
    weights_index: dict[str, Any],
) -> None:
    exported_names = set(weights_index["tensors"])
    original_names = set(state_dict)

    missing = sorted(original_names - exported_names)
    extra = sorted(exported_names - original_names)
    if missing:
        raise ValueError(f"weights.json is missing original tensors: {missing}")
    if extra:
        raise ValueError(f"weights.json contains unknown tensors: {extra}")

    tensor_count = weights_index.get("tensorCount")
    if tensor_count is not None and tensor_count != len(exported_names):
        raise ValueError(
            f"tensorCount is {tensor_count}, but tensors contains {len(exported_names)} entries"
        )


def validate_tensor_bytes(
    name: str,
    tensor: Any,
    tensor_info: dict[str, Any],
    weights_blob: bytes,
) -> None:
    shape = tensor_info.get("shape")
    byte_offset = tensor_info.get("byteOffset")
    byte_length = tensor_info.get("byteLength")

    if not isinstance(shape, list):
        raise ValueError(f"{name} shape must be a list")
    if not isinstance(byte_offset, int) or not isinstance(byte_length, int):
        raise ValueError(f"{name} byteOffset and byteLength must be integers")

    if shape != list(tensor.shape):
        raise ValueError(f"{name} shape mismatch: exported {shape}, original {list(tensor.shape)}")

    expected_byte_length = tensor.numel() * 4
    if byte_length != expected_byte_length:
        raise ValueError(
            f"{name} byteLength is {byte_length}, expected {expected_byte_length}"
        )

    if byte_offset < 0 or byte_length < 0:
        raise ValueError(f"{name} has negative offset or length")
    if byte_offset + byte_length > len(weights_blob):
        raise ValueError(f"{name} points past the end of weights.bin")

    original = tensor.detach().cpu().contiguous().float().numpy()
    original = original.astype("<f4", copy=False)
    restored = np.frombuffer(
        weights_blob,
        dtype="<f4",
        count=byte_length // 4,
        offset=byte_offset,
    ).reshape(shape)

    if not np.array_equal(original, restored):
        difference = np.abs(original - restored)
        max_difference = float(difference.max()) if difference.size else 0.0
        mismatch_count = int(np.count_nonzero(original != restored))
        raise ValueError(
            f"{name} values differ: {mismatch_count} elements mismatch, "
            f"max_abs_diff={max_difference}"
        )


def validate_all_tensors(model: Any, export_dir: Path, weights_index: dict[str, Any]) -> None:
    weights_bin_path = export_dir / "weights.bin"
    validate_file_size(weights_bin_path, weights_index)

    state_dict = model.state_dict()
    validate_tensor_directory(state_dict, weights_index)

    LOGGER.info("Reading raw weights from %s", weights_bin_path)
    weights_blob = weights_bin_path.read_bytes()

    LOGGER.info("Comparing %s tensors", len(state_dict))
    for index, (name, tensor) in enumerate(state_dict.items(), start=1):
        validate_tensor_bytes(name, tensor, weights_index["tensors"][name], weights_blob)
        LOGGER.info("Validated tensor %s/%s: %s", index, len(state_dict), name)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    model_path = args.model_path.resolve() if args.model_path else find_default_model_path()
    export_dir = args.export_dir.resolve() if args.export_dir else find_latest_export_dir()
    local_files_only = not args.allow_download

    weights_index = load_weights_index(export_dir)
    model = load_model(model_path, local_files_only=local_files_only)
    validate_all_tensors(model, export_dir, weights_index)

    LOGGER.info("Tensor validation complete: all exported tensors match exactly")


if __name__ == "__main__":
    main()

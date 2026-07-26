import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("export_raw_gptneo")

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "hf-cache"
DEFAULT_MODEL_CACHE_DIR = CACHE_DIR / "models--roneneldan--TinyStories-3M"
DEFAULT_SNAPSHOTS_DIR = DEFAULT_MODEL_CACHE_DIR / "snapshots"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Hugging Face GPT-Neo causal language model to config.json, "
            "weights.json, and raw little-endian FP32 weights.bin."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to a local Hugging Face model directory. Defaults to the locally "
            "cached roneneldan/TinyStories-3M snapshot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to ../models/output_<YYYYmmdd_HHMMSS> "
            "relative to this script."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download missing model files.",
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


def find_default_model_path() -> Path:
    LOGGER.info("Looking for default TinyStories model snapshot in %s", DEFAULT_SNAPSHOTS_DIR)
    if not DEFAULT_SNAPSHOTS_DIR.exists():
        raise FileNotFoundError(
            "TinyStories model is not present locally. Run `python run_tinystories.py` "
            "first, or pass --model-path."
        )

    for snapshot in sorted(DEFAULT_SNAPSHOTS_DIR.iterdir(), reverse=True):
        if (snapshot / "config.json").exists():
            LOGGER.info("Using default model snapshot: %s", snapshot)
            return snapshot

    raise FileNotFoundError(
        "TinyStories model cache is incomplete. Run `python run_tinystories.py` first, "
        "or pass --model-path."
    )


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_DIR.parent / "models" / f"output_{timestamp}"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def get_config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def build_export_config(model: Any) -> dict[str, Any]:
    config = model.config
    hidden_size = get_config_value(config, "hidden_size")
    num_heads = get_config_value(config, "num_heads")
    intermediate_size = get_config_value(config, "intermediate_size")
    if intermediate_size is None and hidden_size is not None:
        intermediate_size = hidden_size * 4

    export_config = {
        "architecture": get_config_value(config, "model_type"),
        "vocabularySize": get_config_value(config, "vocab_size"),
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate_size,
        "numberOfLayers": get_config_value(config, "num_layers"),
        "numberOfHeads": num_heads,
        "headDimension": hidden_size // num_heads if hidden_size and num_heads else None,
        "maximumSequenceLength": get_config_value(config, "max_position_embeddings"),
        "layerNormEpsilon": get_config_value(config, "layer_norm_epsilon"),
        "activation": get_config_value(config, "activation_function"),
        "tiedWordEmbeddings": get_config_value(config, "tie_word_embeddings"),
        "attentionLayers": get_config_value(config, "attention_layers"),
        "attentionTypes": get_config_value(config, "attention_types"),
        "windowSize": get_config_value(config, "window_size"),
        "bosTokenId": get_config_value(config, "bos_token_id"),
        "eosTokenId": get_config_value(config, "eos_token_id"),
        "padTokenId": get_config_value(config, "pad_token_id"),
    }
    return {key: jsonable(value) for key, value in export_config.items()}


def write_json(path: Path, data: dict[str, Any]) -> None:
    LOGGER.info("Writing %s", path)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def tensor_to_little_endian_fp32_bytes(tensor: Any) -> bytes:
    import torch

    array = tensor.detach().cpu().contiguous().to(torch.float32).numpy()
    little_endian_array = array.astype("<f4", copy=False)
    return little_endian_array.tobytes(order="C")


def export_weights(model: Any, weights_bin_path: Path) -> dict[str, Any]:
    state_dict = model.state_dict()
    tensor_count = len(state_dict)
    LOGGER.info("Exporting %s tensors to %s", tensor_count, weights_bin_path)

    weights_index: dict[str, Any] = {
        "dtype": "float32",
        "byteOrder": "little-endian",
        "totalByteLength": 0,
        "tensorCount": tensor_count,
        "tensors": {},
    }

    byte_offset = 0
    with weights_bin_path.open("wb") as weights_file:
        for index, (name, tensor) in enumerate(state_dict.items(), start=1):
            data = tensor_to_little_endian_fp32_bytes(tensor)
            byte_length = len(data)
            expected_byte_length = tensor.numel() * 4
            if byte_length != expected_byte_length:
                raise ValueError(
                    f"{name} produced {byte_length} bytes, expected {expected_byte_length}"
                )

            weights_file.write(data)
            weights_index["tensors"][name] = {
                "shape": list(tensor.shape),
                "byteOffset": byte_offset,
                "byteLength": byte_length,
            }

            LOGGER.info(
                "Exported tensor %s/%s: %s shape=%s offset=%s bytes=%s",
                index,
                tensor_count,
                name,
                list(tensor.shape),
                byte_offset,
                byte_length,
            )
            byte_offset += byte_length

    weights_index["totalByteLength"] = byte_offset
    actual_size = weights_bin_path.stat().st_size
    if actual_size != byte_offset:
        raise ValueError(
            f"{weights_bin_path} is {actual_size} bytes, expected {byte_offset} bytes"
        )

    LOGGER.info("Finished weights.bin: %s bytes", actual_size)
    return weights_index


def load_model(model_path: Path, local_files_only: bool) -> Any:
    from transformers import AutoModelForCausalLM

    LOGGER.info("Loading model from %s", model_path)
    LOGGER.info("local_files_only=%s", local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    model.eval()
    LOGGER.info("Loaded model class: %s", model.__class__.__name__)
    LOGGER.info("Loaded model type: %s", getattr(model.config, "model_type", None))
    return model


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    model_path = args.model_path.resolve() if args.model_path else find_default_model_path()
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir()
    local_files_only = not args.allow_download

    LOGGER.info("Creating output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    model = load_model(model_path, local_files_only=local_files_only)

    config_json = build_export_config(model)
    write_json(output_dir / "config.json", config_json)

    weights_json = export_weights(model, output_dir / "weights.bin")
    write_json(output_dir / "weights.json", weights_json)

    LOGGER.info("Export complete")
    LOGGER.info("config.json: %s", output_dir / "config.json")
    LOGGER.info("weights.json: %s", output_dir / "weights.json")
    LOGGER.info("weights.bin: %s", output_dir / "weights.bin")


if __name__ == "__main__":
    main()

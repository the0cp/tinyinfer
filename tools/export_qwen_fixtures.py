#!/usr/bin/env python3
"""Export deterministic Qwen2.5 reference data for tinyinfer.

The script intentionally supports one pinned model. It validates the real
checkpoint contract before asking Hugging Face Transformers for tokenizer,
prefill, and cached-decode reference values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHTS_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"

EXPECTED_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_sliding_window": False,
    "vocab_size": 151936,
}

CHAT_MESSAGES = [
    {"role": "system", "content": "You are a concise inference systems tutor."},
    {"role": "user", "content": "用一句话解释 KV cache。"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local snapshot of the pinned Hugging Face model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for fixtures.json and fixtures.bin.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3,
        choices=range(1, 9),
        metavar="1..8",
        help="Fixed number of greedy reference decode steps (default: 3).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tensors(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    query_size = config["num_attention_heads"] * (hidden // config["num_attention_heads"])
    kv_size = config["num_key_value_heads"] * (hidden // config["num_attention_heads"])

    tensors: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config["vocab_size"], hidden),
        "model.norm.weight": (hidden,),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (query_size, hidden),
                f"{prefix}.self_attn.q_proj.bias": (query_size,),
                f"{prefix}.self_attn.k_proj.weight": (kv_size, hidden),
                f"{prefix}.self_attn.k_proj.bias": (kv_size,),
                f"{prefix}.self_attn.v_proj.weight": (kv_size, hidden),
                f"{prefix}.self_attn.v_proj.bias": (kv_size,),
                f"{prefix}.self_attn.o_proj.weight": (hidden, query_size),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return tensors


def validate_source(model_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    required_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing_files = [name for name in required_files if not (model_dir / name).is_file()]
    if missing_files:
        raise ValueError(f"model directory is missing: {', '.join(missing_files)}")

    config = load_json(model_dir / "config.json")
    mismatches = [
        f"{name}: expected {expected!r}, got {config.get(name)!r}"
        for name, expected in EXPECTED_CONFIG.items()
        if config.get(name) != expected
    ]
    if mismatches:
        raise ValueError("unsupported Qwen config:\n  " + "\n  ".join(mismatches))

    weights_path = model_dir / "model.safetensors"
    weights_hash = sha256_file(weights_path)
    if weights_hash != WEIGHTS_SHA256:
        raise ValueError(
            f"model.safetensors SHA-256 mismatch: expected {WEIGHTS_SHA256}, got {weights_hash}"
        )

    expected = expected_tensors(config)
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        actual_names = set(checkpoint.keys())
        missing = sorted(set(expected) - actual_names)
        unexpected = sorted(actual_names - set(expected))
        shape_errors = []
        dtype_errors = []
        for name, shape in expected.items():
            if name not in actual_names:
                continue
            tensor = checkpoint.get_slice(name)
            if tuple(tensor.get_shape()) != shape:
                shape_errors.append(f"{name}: expected {shape}, got {tuple(tensor.get_shape())}")
            if tensor.get_dtype() != "BF16":
                dtype_errors.append(f"{name}: expected BF16, got {tensor.get_dtype()}")

    errors = []
    if missing:
        errors.append("missing tensors: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected tensors: " + ", ".join(unexpected))
    errors.extend(shape_errors)
    errors.extend(dtype_errors)
    if errors:
        raise ValueError("unsupported checkpoint:\n  " + "\n  ".join(errors))

    source_hashes = {"model.safetensors": weights_hash}
    source_hashes.update(
        {
            name: sha256_file(model_dir / name)
            for name in required_files
            if name != "model.safetensors"
        }
    )
    return config, source_hashes


def as_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def add_array(arrays: dict[str, np.ndarray], name: str, value: Any) -> None:
    if name in arrays:
        raise ValueError(f"duplicate fixture array: {name}")
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype("<f4", copy=False)
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype("<i8", copy=False)
    else:
        raise ValueError(f"unsupported fixture dtype for {name}: {array.dtype}")
    arrays[name] = np.ascontiguousarray(array)


def export_token_cases(tokenizer: Any) -> tuple[list[dict[str, Any]], torch.Tensor]:
    chat_text = tokenizer.apply_chat_template(
        CHAT_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )
    cases = [
        ("plain_ascii", "Hello, TinyInfer!"),
        ("plain_utf8", "你好，KV 缓存是什么？"),
        ("chat", chat_text),
    ]

    exported = []
    chat_ids = None
    for name, text in cases:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        exported.append(
            {
                "name": name,
                "text": text,
                "token_ids": token_ids,
                "decoded": tokenizer.decode(token_ids, skip_special_tokens=False),
            }
        )
        if name == "chat":
            chat_ids = torch.tensor([token_ids], dtype=torch.long)

    assert chat_ids is not None
    return exported, chat_ids


def export_model_arrays(
    model_dir: Path,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    arrays: dict[str, np.ndarray] = {}
    add_array(arrays, "input_ids", input_ids.numpy())

    with torch.inference_mode():
        hidden = model.model.embed_tokens(input_ids)
        layer = model.model.layers[0]
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn

        batch, tokens, _ = normalized.shape
        queries = attention.q_proj(normalized).view(
            batch, tokens, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        keys = attention.k_proj(normalized).view(
            batch, tokens, attention.num_key_value_heads, attention.head_dim
        ).transpose(1, 2)
        values = attention.v_proj(normalized).view(
            batch, tokens, attention.num_key_value_heads, attention.head_dim
        ).transpose(1, 2)
        positions = torch.arange(tokens, dtype=torch.long).unsqueeze(0)
        cos, sin = attention.rotary_emb(values, seq_len=tokens)
        rotated_queries, rotated_keys = apply_rotary_pos_emb(
            queries, keys, cos, sin, positions
        )

        outputs = model(
            input_ids=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        add_array(arrays, "embedding", as_array(hidden[0]))
        add_array(arrays, "layer0_input_norm", as_array(normalized[0]))
        add_array(arrays, "layer0_query", as_array(queries.transpose(1, 2)[0]))
        add_array(arrays, "layer0_key", as_array(keys.transpose(1, 2)[0]))
        add_array(arrays, "layer0_value", as_array(values.transpose(1, 2)[0]))
        add_array(arrays, "layer0_query_rope", as_array(rotated_queries.transpose(1, 2)[0]))
        add_array(arrays, "layer0_key_rope", as_array(rotated_keys.transpose(1, 2)[0]))
        add_array(arrays, "layer0_output", as_array(outputs.hidden_states[1][0]))
        add_array(arrays, "final_hidden", as_array(outputs.hidden_states[-1][0]))

        cache_key, cache_value = outputs.past_key_values[0]
        torch.testing.assert_close(cache_key, rotated_keys, rtol=0, atol=0)
        torch.testing.assert_close(cache_value, values, rtol=0, atol=0)
        add_array(arrays, "layer0_cache_key", as_array(cache_key.transpose(1, 2)[0]))
        add_array(arrays, "layer0_cache_value", as_array(cache_value.transpose(1, 2)[0]))

        logits = outputs.logits[:, -1, :]
        add_array(arrays, "prefill_last_logits", as_array(logits[0]))
        past = outputs.past_key_values
        generated: list[int] = []

        for step in range(max_new_tokens):
            token = torch.argmax(logits, dim=-1)
            generated.append(int(token.item()))
            if step + 1 == max_new_tokens:
                break
            outputs = model(
                input_ids=token[:, None],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            add_array(arrays, f"decode_{step + 1}_logits", as_array(logits[0]))

    del model
    return arrays, generated


def write_fixtures(
    output_dir: Path,
    config: dict[str, Any],
    source_hashes: dict[str, str],
    token_cases: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    generated: list[int],
    generated_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "fixtures.bin"
    entries = []
    offset = 0

    with binary_path.open("wb") as stream:
        for name, array in arrays.items():
            payload = array.tobytes(order="C")
            stream.write(payload)
            entries.append(
                {
                    "name": name,
                    "dtype": "f32" if array.dtype.kind == "f" else "i64",
                    "shape": list(array.shape),
                    "offset_bytes": offset,
                    "byte_size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            offset += len(payload)

    manifest = {
        "format": "tinyinfer_qwen_fixture",
        "version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "config": {name: config[name] for name in EXPECTED_CONFIG},
            "source_sha256": source_hashes,
        },
        "reference": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "threads": 1,
        },
        "token_cases": token_cases,
        "generation": {
            "strategy": "greedy_argmax",
            "token_ids": generated,
            "decoded": generated_text,
        },
        "binary": {
            "file": binary_path.name,
            "byte_size": offset,
            "sha256": sha256_file(binary_path),
        },
        "arrays": entries,
    }
    manifest_path = output_dir / "fixtures.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()

    config, source_hashes = validate_source(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    token_cases, input_ids = export_token_cases(tokenizer)
    arrays, generated = export_model_arrays(model_dir, input_ids, args.max_new_tokens)
    generated_text = tokenizer.decode(generated, skip_special_tokens=False)
    write_fixtures(
        output_dir,
        config,
        source_hashes,
        token_cases,
        arrays,
        generated,
        generated_text,
    )
    print(f"wrote {output_dir / 'fixtures.json'}")
    print(f"wrote {output_dir / 'fixtures.bin'}")


if __name__ == "__main__":
    main()

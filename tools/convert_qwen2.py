#!/usr/bin/env python3
"""Convert the pinned Qwen2.5 checkpoint into tinyinfer's BF16 package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from export_qwen_fixtures import MODEL_ID, REVISION, validate_source


ALIGNMENT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def tensor_specs(config: dict[str, object]) -> Iterator[tuple[str, bool]]:
    yield "model.embed_tokens.weight", False
    layers = int(config["num_hidden_layers"])
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        yield f"{prefix}.input_layernorm.weight", False
        yield f"{prefix}.self_attn.q_proj.weight", True
        yield f"{prefix}.self_attn.q_proj.bias", False
        yield f"{prefix}.self_attn.k_proj.weight", True
        yield f"{prefix}.self_attn.k_proj.bias", False
        yield f"{prefix}.self_attn.v_proj.weight", True
        yield f"{prefix}.self_attn.v_proj.bias", False
        yield f"{prefix}.self_attn.o_proj.weight", True
        yield f"{prefix}.post_attention_layernorm.weight", False
        yield f"{prefix}.mlp.gate_proj.weight", True
        yield f"{prefix}.mlp.up_proj.weight", True
        yield f"{prefix}.mlp.down_proj.weight", True
    yield "model.norm.weight", False


def align(offset: int) -> int:
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "model.tiq"
    weights_path = output_dir / "weights.bin"
    tokenizer_paths = [output_dir / "tokenizer.json", output_dir / "tokenizer_config.json"]
    outputs = [manifest_path, weights_path, *tokenizer_paths]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    config, source_hashes = validate_source(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, tuple[int, ...], int, int]] = []
    offset = 0
    parameter_count = 0
    source_path = model_dir / "model.safetensors"
    with safe_open(source_path, framework="pt", device="cpu") as checkpoint:
        with weights_path.open("wb") as weights:
            for name, transpose in tensor_specs(config):
                tensor = checkpoint.get_tensor(name)
                if transpose:
                    tensor = tensor.transpose(0, 1)
                tensor = tensor.contiguous()
                raw = tensor.view(torch.uint16).numpy()

                aligned = align(offset)
                weights.write(b"\0" * (aligned - offset))
                payload = raw.astype("<u2", copy=False).tobytes(order="C")
                weights.write(payload)
                entries.append((name, tuple(tensor.shape), aligned, len(payload)))
                offset = aligned + len(payload)
                parameter_count += tensor.numel()

    lines = [
        "TINYINFER_QWEN2 1",
        "weights weights.bin",
        f"model_id {MODEL_ID}",
        f"revision {REVISION}",
        f"source_sha256 {source_hashes['model.safetensors']}",
        "model_type qwen2",
        "hidden_act silu",
        "rope_layout half_split",
        "qkv_bias 1",
        "tie_word_embeddings 1",
        f"alignment {ALIGNMENT}",
        f"vocab_size {config['vocab_size']}",
        f"hidden_size {config['hidden_size']}",
        f"intermediate_size {config['intermediate_size']}",
        f"num_layers {config['num_hidden_layers']}",
        f"query_heads {config['num_attention_heads']}",
        f"kv_heads {config['num_key_value_heads']}",
        f"max_position_embeddings {config['max_position_embeddings']}",
        f"rms_norm_epsilon {float(config['rms_norm_eps']):.9g}",
        f"rope_theta {float(config['rope_theta']):.9g}",
        f"bos_token_id {config['bos_token_id']}",
        f"eos_token_id {config['eos_token_id']}",
    ]
    for name, shape, tensor_offset, byte_size in entries:
        dimensions = " ".join(str(value) for value in shape)
        lines.append(
            f"tensor {name} bf16 {len(shape)} {dimensions} "
            f"{tensor_offset} {byte_size}"
        )
    lines.append("end")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    shutil.copyfile(model_dir / "tokenizer.json", tokenizer_paths[0])
    shutil.copyfile(model_dir / "tokenizer_config.json", tokenizer_paths[1])
    print(f"wrote {manifest_path}")
    print(f"wrote {weights_path} ({offset} bytes, {parameter_count} stored parameters)")


if __name__ == "__main__":
    main()

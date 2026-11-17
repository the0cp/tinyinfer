#!/usr/bin/env python3
"""Run tinyinfer evaluation benchmarks and render reusable figures."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from statistics import median

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
SHORT_NAMES = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot",
)
COLORS = ("#2563EB", "#F59E0B", "#10B981", "#8B5CF6")
REFERENCE_SYSTEM_PROMPT = "You are a concise inference systems tutor."
REFERENCE_USER_PROMPT = "Explain KV cache in one short sentence."
CLI_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    default_models = Path.home() / "tinyinfer-models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-release")
    parser.add_argument(
        "--fashion-model",
        type=Path,
        default=default_models / "fashion-mnist-final",
    )
    parser.add_argument(
        "--fashion-data",
        type=Path,
        default=default_models / "fashion-mnist-data",
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=default_models / "qwen2.5-0.5b",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "evaluation",
    )
    parser.add_argument(
        "--copy-dir",
        action="append",
        type=Path,
        default=[],
        help="copy rendered PNG and SVG files to this directory",
    )
    parser.add_argument("--qwen-runs", type=int, default=5)
    parser.add_argument(
        "--max-new-tokens",
        "--generated-tokens",
        dest="max_new_tokens",
        type=int,
        default=64,
        help="generation safety limit; generation may stop earlier at EOS",
    )
    return parser.parse_args()


def run(*command: object) -> str:
    result = subprocess.run(
        [str(value) for value in command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def csv_output(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text.strip())))


def values(text: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z][A-Za-z0-9_]*)=([^,\s]+)", text))


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(
            r"^model name\s*:\s*(.+)$",
            cpuinfo.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return platform.processor() or platform.machine()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    return np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(-1, 784)


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    return np.frombuffer(payload, dtype=np.uint8, offset=8).astype(np.int64)


def load_fashion_weights(model_dir: Path) -> tuple[np.ndarray, ...]:
    manifest = model_dir / "model.ti"
    weight_file: Path | None = None
    tensors: dict[str, np.ndarray] = {}

    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "weights":
            weight_file = manifest.parent / fields[1]
        if fields[0] != "tensor":
            continue

        name = fields[1]
        if fields[2] != "f32":
            raise ValueError(f"unsupported Fashion-MNIST tensor type: {fields[2]}")
        rank = int(fields[3])
        shape = tuple(int(value) for value in fields[4 : 4 + rank])
        offset = int(fields[4 + rank])
        count = int(np.prod(shape))
        tensors[name] = np.fromfile(
            weight_file or model_dir / "weights.bin",
            dtype="<f4",
            count=count,
            offset=offset,
        ).reshape(shape)

    return (
        tensors["fc1.weight"],
        tensors["fc1.bias"],
        tensors["fc2.weight"],
        tensors["fc2.bias"],
    )


def eval_fashion(
    model_dir: Path,
    data_dir: Path,
    tinyinfer: Path,
) -> dict[str, object]:
    images = read_images(data_dir / "t10k-images-idx3-ubyte.gz")
    labels = read_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    weight1, bias1, weight2, bias2 = load_fashion_weights(model_dir)
    predictions = np.empty(labels.size, dtype=np.int64)

    for begin in range(0, labels.size, 1024):
        inputs = images[begin : begin + 1024].astype(np.float32) / 255.0
        hidden = np.maximum(inputs @ weight1 + bias1, 0.0)
        predictions[begin : begin + inputs.shape[0]] = np.argmax(
            hidden @ weight2 + bias2,
            axis=1,
        )

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = true_positive / support
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall != 0,
    )

    runtime = run(
        tinyinfer,
        "run",
        model_dir / "model.ti",
        model_dir / "sample.f32",
    )
    output_line = next(
        line for line in runtime.splitlines() if line.startswith("output_values=")
    )
    actual = np.fromstring(output_line.partition("=")[2], sep=" ", dtype=np.float32)
    expected = np.fromfile(model_dir / "expected.f32", dtype="<f4")
    runtime_values = values(runtime)

    per_class = [
        {
            "class": name,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    ]
    confusion_rows = [
        {
            "true_class": CLASS_NAMES[truth],
            "predicted_class": CLASS_NAMES[prediction],
            "count": int(confusion[truth, prediction]),
        }
        for truth in range(len(CLASS_NAMES))
        for prediction in range(len(CLASS_NAMES))
    ]
    return {
        "samples": int(labels.size),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1)),
        "runtime_sample_max_abs_error": float(np.max(np.abs(actual - expected))),
        "runtime_sample_inference_ms": float(runtime_values["inference_ms"]),
        "runtime_sample_peak_live_bytes": int(runtime_values["peak_live_bytes"]),
        "per_class": per_class,
        "confusion": confusion_rows,
    }


def eval_qwen(
    model_dir: Path,
    checker: Path,
    tinyinfer: Path,
    runs: int,
    max_new_tokens: int,
) -> dict[str, object]:
    check = run(
        checker,
        model_dir / "model.tiq",
        model_dir / "tokenizer.json",
        model_dir / "model_case.bin",
    )
    metrics_text = check.split("layer_metrics_begin\n", maxsplit=1)[1].split(
        "layer_metrics_end\n",
        maxsplit=1,
    )[0]
    layer_metrics = [
        {
            "phase": row["phase"],
            "stage": row["stage"],
            "max_abs_error": float(row["max_abs_error"]),
            "mean_abs_error": float(row["mean_abs_error"]),
            "cosine_similarity": float(row["cosine_similarity"]),
        }
        for row in csv_output(metrics_text)
    ]
    correctness = [
        {
            "step": 0 if row["phase"] == "prefill" else int(row["phase"][7:]),
            "phase": row["phase"],
            "max_abs_error": row["max_abs_error"],
            "mean_abs_error": row["mean_abs_error"],
            "cosine_similarity": row["cosine_similarity"],
        }
        for row in layer_metrics
        if row["stage"] == "Logits"
    ]

    fixture = json.loads((model_dir / "fixtures.json").read_text(encoding="utf-8"))
    chat = next(case for case in fixture["token_cases"] if case["name"] == "chat")
    reference = {
        "system_prompt": REFERENCE_SYSTEM_PROMPT,
        "user_prompt": REFERENCE_USER_PROMPT,
        "rendered_prompt": chat["text"],
        "prompt_tokens": len(chat["token_ids"]),
        "generated_tokens": len(fixture["generation"]["token_ids"]),
        "decode_steps": max(0, len(fixture["generation"]["token_ids"]) - 1),
        "max_new_tokens": fixture["generation"]["max_new_tokens"],
        "stop_reason": fixture["generation"]["stop_reason"],
        "generated_text": fixture["generation"]["decoded"].removesuffix("<|im_end|>"),
        "layers": fixture["model_case"]["layers"],
    }

    performance: list[dict[str, object]] = []
    for index in range(runs):
        output = run(
            tinyinfer,
            "generate",
            model_dir,
            REFERENCE_USER_PROMPT,
            max_new_tokens,
        )
        result = values(output)
        decoded = output.split("\n\n", maxsplit=1)[0].strip()
        generated = int(result["generated_tokens"])
        decode_steps = max(0, generated - 1)
        decode_ms = float(result["decode_ms"])
        performance.append(
            {
                "run": index + 1,
                "prompt_tokens": int(result["prompt_tokens"]),
                "generated_tokens": generated,
                "prefill_ms": float(result["prefill_ms"]),
                "decode_ms": decode_ms,
                "decode_ms_per_token": decode_ms / decode_steps if decode_steps else 0.0,
                "decode_tokens_per_second": float(
                    result["decode_tokens_per_second"]
                ),
                "generated_tokens_per_second": float(
                    result["generated_tokens_per_second"]
                ),
                "text": decoded,
            }
        )

    return {
        "reference": reference,
        "layer_metrics": layer_metrics,
        "correctness": correctness,
        "performance_prompt": {
            "system_prompt": CLI_SYSTEM_PROMPT,
            "user_prompt": REFERENCE_USER_PROMPT,
            "max_new_tokens": max_new_tokens,
        },
        "performance": performance,
    }


def eval_benchmarks(build_dir: Path) -> dict[str, object]:
    kv_cache = csv_output(run(build_dir / "bench_kv_cache"))
    paged_output = run(build_dir / "bench_paged_kv_cache")
    paged_text, prefix_text = paged_output.strip().split("\n\n", maxsplit=1)
    linear_output = run(build_dir / "bench_inference_linear")
    linear = []
    for line in linear_output.splitlines():
        if " serial_ms=" not in line:
            continue
        name, metrics = line.split(maxsplit=1)
        row = values(metrics)
        linear.append(
            {
                "case": name,
                "serial_ms": float(row["serial_ms"]),
                "threaded_ms": float(row["threaded_ms"]),
                "speedup": float(row["speedup"]),
                "selected": row["selected"],
            }
        )

    memory_runs = [
        values(run(build_dir / "bench_memory_plan"))
        for _ in range(5)
    ]
    memory = dict(memory_runs[0])
    for key in (
        "unplanned_batch_mean_median_us",
        "unplanned_batch_mean_p95_us",
        "planned_batch_mean_median_us",
        "planned_batch_mean_p95_us",
    ):
        memory[key] = str(median(float(row[key]) for row in memory_runs))
    return {
        "kv_cache": kv_cache,
        "paged_kv_cache": csv_output(paged_text),
        "prefix_sharing": csv_output(prefix_text),
        "linear": linear,
        "memory_plan": memory,
        "memory_plan_runs": memory_runs,
    }


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def save(
    figure: plt.Figure,
    name: str,
    figure_dir: Path,
    copy_dirs: list[Path],
) -> None:
    figure.tight_layout()
    outputs = []
    for suffix, options in (("png", {"dpi": 200}), ("svg", {})):
        path = figure_dir / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(figure)

    for directory in copy_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            shutil.copy2(path, directory / path.name)


def plot_fashion(
    result: dict[str, object],
    figure_dir: Path,
    copy_dirs: list[Path],
) -> None:
    matrix = np.zeros((10, 10), dtype=np.float64)
    for row in result["confusion"]:
        truth = CLASS_NAMES.index(row["true_class"])
        prediction = CLASS_NAMES.index(row["predicted_class"])
        matrix[truth, prediction] = row["count"]
    matrix /= matrix.sum(axis=1, keepdims=True)

    figure, axis = plt.subplots(figsize=(8.4, 7.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.grid(False)
    axis.set_title(
        "Fashion-MNIST confusion matrix\n"
        f"Accuracy {result['accuracy']:.1%} · Macro F1 {result['macro_f1']:.1%}"
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(10), SHORT_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(10), SHORT_NAMES)
    for row in range(10):
        for column in range(10):
            if matrix[row, column] >= 0.01:
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > 0.55 else "#0F172A",
                    fontsize=8,
                )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Share within each true class")
    save(figure, "fashion_confusion_matrix", figure_dir, copy_dirs)

    classes = result["per_class"]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    positions = np.arange(len(classes))
    bars = axis.barh(
        positions,
        [row["f1"] for row in classes],
        color=COLORS[0],
    )
    axis.set_yticks(positions, SHORT_NAMES)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("F1 score")
    axis.set_title("Fashion-MNIST per-class F1 score")
    axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    save(figure, "fashion_class_f1", figure_dir, copy_dirs)


def plot_qwen(
    result: dict[str, object],
    figure_dir: Path,
    copy_dirs: list[Path],
) -> None:
    correctness = result["correctness"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    steps = [row["step"] for row in correctness]
    axis.plot(
        steps,
        [row["max_abs_error"] for row in correctness],
        marker="o",
        linewidth=2.2,
        label="Maximum absolute error",
        color=COLORS[0],
    )
    axis.plot(
        steps,
        [row["mean_abs_error"] for row in correctness],
        marker="o",
        linewidth=2.2,
        label="Mean absolute error",
        color=COLORS[1],
    )
    axis.set_yscale("log")
    tick_stride = max(1, len(steps) // 8)
    ticks = steps[::tick_stride]
    if ticks[-1] != steps[-1]:
        ticks.append(steps[-1])
    axis.set_xticks(
        ticks,
        ["Prefill" if step == 0 else f"D{step}" for step in ticks],
    )
    axis.set_xlabel("Generation phase")
    axis.set_ylabel("Absolute logit error (log scale)")
    axis.set_title("Final-logit error across Prefill and cached Decode")
    axis.legend()
    save(figure, "qwen_logit_error", figure_dir, copy_dirs)

    phases = ["prefill"] + [f"decode_{step}" for step in steps[1:]]
    stages = ["Embedding"] + [
        f"Layer {layer}" for layer in range(result["reference"]["layers"])
    ] + ["Final RMSNorm"]
    rows = {
        (row["stage"], row["phase"]): row
        for row in result["layer_metrics"]
        if row["stage"] != "Logits"
    }
    for field, title, name, color_label in (
        (
            "max_abs_error",
            "Maximum absolute error by Qwen layer and generation phase",
            "qwen_layer_max_error",
            "Maximum absolute error (×10⁻⁶)",
        ),
        (
            "mean_abs_error",
            "Mean absolute error by Qwen layer and generation phase",
            "qwen_layer_mean_error",
            "Mean absolute error (×10⁻⁶)",
        ),
    ):
        matrix = np.array(
            [
                [rows[(stage, phase)][field] * 1_000_000.0 for phase in phases]
                for stage in stages
            ]
        )
        width = min(18.0, max(10.0, 5.0 + 0.32 * len(phases)))
        figure, axis = plt.subplots(figsize=(width, 9.0))
        image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0)
        axis.grid(False)
        x_stride = max(1, len(phases) // 12)
        x_positions = list(range(0, len(phases), x_stride))
        if x_positions[-1] != len(phases) - 1:
            x_positions.append(len(phases) - 1)
        axis.set_xticks(
            x_positions,
            ["Prefill" if index == 0 else f"D{index}" for index in x_positions],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(stages)), stages)
        axis.set_xlabel("Generation phase")
        axis.set_ylabel("Compared output")
        axis.set_title(title)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
        colorbar.set_label(color_label)
        save(figure, name, figure_dir, copy_dirs)

    performance = result["performance"]
    prefill = [row["prefill_ms"] for row in performance]
    per_token = [row["decode_ms_per_token"] for row in performance]
    medians = [median(prefill), median(per_token)]
    lower = [medians[0] - min(prefill), medians[1] - min(per_token)]
    upper = [max(prefill) - medians[0], max(per_token) - medians[1]]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    bars = axis.bar(
        ["Prefill total", "Decode per token"],
        medians,
        yerr=np.array([lower, upper]),
        capsize=6,
        color=(COLORS[0], COLORS[1]),
    )
    axis.set_ylabel("Latency (ms)")
    axis.set_title(
        f"Qwen2.5-0.5B generation latency · median of {len(performance)} runs\n"
        "Whiskers show the observed minimum–maximum range"
    )
    axis.bar_label(bars, fmt="%.1f ms", padding=5)
    throughput = median(
        [row["decode_tokens_per_second"] for row in performance]
    )
    axis.text(
        0.98,
        0.95,
        f"Median decode throughput: {throughput:.2f} tokens/s",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#334155",
    )
    save(figure, "qwen_generation_latency", figure_dir, copy_dirs)


def plot_benchmarks(
    result: dict[str, object],
    figure_dir: Path,
    copy_dirs: list[Path],
) -> None:
    kv_rows = result["kv_cache"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for index, kv_heads in enumerate(("8", "2", "1")):
        rows = [row for row in kv_rows if row["kv_heads"] == kv_heads]
        axis.plot(
            [int(row["tokens"]) for row in rows],
            [float(row["speedup"]) for row in rows],
            marker="o",
            linewidth=2.2,
            color=COLORS[index],
            label={"8": "MHA: 8 KV heads", "2": "GQA: 2 KV heads", "1": "MQA: 1 KV head"}[kv_heads],
        )
    axis.set_xticks((32, 64, 96))
    axis.set_xlabel("Total sequence length (tokens)")
    axis.set_ylabel("Speedup over full recomputation")
    axis.set_title("KV cache avoids repeated attention work")
    axis.legend()
    save(figure, "kv_cache_speedup", figure_dir, copy_dirs)

    paged = result["paged_kv_cache"]
    block_sizes = [int(row["block_tokens"]) for row in paged]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.plot(
        block_sizes,
        [float(row["paged_utilization"]) * 100.0 for row in paged],
        marker="o",
        linewidth=2.2,
        color=COLORS[0],
        label="Mapped-block utilization",
    )
    axis.plot(
        block_sizes,
        [float(row["saving_vs_contiguous"]) * 100.0 for row in paged],
        marker="o",
        linewidth=2.2,
        color=COLORS[2],
        label="Memory reduction vs max reservation",
    )
    axis.set_xticks(block_sizes)
    axis.set_ylim(0.0, 105.0)
    axis.set_xlabel("Tokens per block")
    axis.set_ylabel("Percent")
    axis.set_title("Paged KV cache block-size trade-off")
    axis.legend()
    save(figure, "paged_kv_memory", figure_dir, copy_dirs)

    linear = result["linear"]
    labels = [row["case"].replace("_", "\n") for row in linear]
    speedups = [row["speedup"] for row in linear]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.bar(labels, speedups, color=COLORS[0])
    axis.axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.4)
    axis.set_ylabel("Serial time / threaded time")
    axis.set_title("Linear threading helps only when work is large enough")
    axis.bar_label(bars, fmt="%.2f×", padding=4)
    save(figure, "linear_threading_speedup", figure_dir, copy_dirs)

    memory = result["memory_plan"]
    groups = ("Median", "P95")
    unplanned = (
        float(memory["unplanned_batch_mean_median_us"]),
        float(memory["unplanned_batch_mean_p95_us"]),
    )
    planned = (
        float(memory["planned_batch_mean_median_us"]),
        float(memory["planned_batch_mean_p95_us"]),
    )
    positions = np.arange(len(groups))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    left = axis.bar(positions - width / 2, unplanned, width, label="Without plan", color="#94A3B8")
    right = axis.bar(positions + width / 2, planned, width, label="With reuse plan", color=COLORS[2])
    axis.set_xticks(positions, groups)
    axis.set_ylabel("Batch-mean latency (µs)")
    axis.set_title(
        "Memory reuse lowers latency\n"
        f"Managed buffers: {memory['unplanned_managed_buffers']} → "
        f"{memory['planned_managed_buffers']}"
    )
    axis.legend()
    axis.bar_label(left, fmt="%.2f", padding=3)
    axis.bar_label(right, fmt="%.2f", padding=3)
    save(figure, "memory_plan_latency", figure_dir, copy_dirs)


def write_report(path: Path, result: dict[str, object]) -> None:
    fashion = result["fashion_mnist"]
    qwen = result["qwen"]
    qwen_perf = qwen["performance"]
    reference = qwen["reference"]
    performance_prompt = qwen["performance_prompt"]
    max_error = max(row["max_abs_error"] for row in qwen["correctness"])
    throughput = median([row["decode_tokens_per_second"] for row in qwen_perf])
    memory = result["benchmarks"]["memory_plan"]
    path.write_text(
        "\n".join(
            (
                "# Evaluation snapshot",
                "",
                "The figures in this directory are generated from the current Release build. "
                "Raw values are available in the adjacent CSV and JSON files.",
                "",
                "## Results",
                "",
                f"- Fashion-MNIST: {fashion['accuracy']:.2%} accuracy and "
                f"{fashion['macro_f1']:.2%} macro F1 on {fashion['samples']:,} test images.",
                f"- Fashion runtime check: maximum logit error "
                f"{fashion['runtime_sample_max_abs_error']:.3g} on the exported sample.",
                f"- Qwen reference check: all greedy tokens match; maximum observed logit "
                f"error {max_error:.3g}.",
                f"- Qwen reference generation: {reference['prompt_tokens']} prompt tokens, "
                f"{reference['generated_tokens']} generated token ids including EOS, "
                f"{reference['decode_steps']} cached Decode steps, stop={reference['stop_reason']}.",
                f"- Qwen reference output: `{reference['generated_text']}`",
                f"- Qwen generation: median decode throughput {throughput:.2f} tokens/s "
                f"across {len(qwen_perf)} runs.",
                f"- Memory plan: median batch-mean latency "
                f"{float(memory['unplanned_batch_mean_median_us']):.2f} → "
                f"{float(memory['planned_batch_mean_median_us']):.2f} µs.",
                "",
                "## Method",
                "",
                f"- Correctness prompt: system=`{reference['system_prompt']}`; "
                f"user=`{reference['user_prompt']}`.",
                f"- Correctness generation stops at EOS; {reference['max_new_tokens']} is only "
                "the safety limit.",
                f"- Correctness compares Embedding, all {reference['layers']} decoder layers, "
                "Final RMSNorm and final logits for Prefill and every cached Decode step.",
                f"- Performance prompt: system=`{performance_prompt['system_prompt']}`; "
                f"user=`{performance_prompt['user_prompt']}`; safety limit="
                f"{performance_prompt['max_new_tokens']} new tokens.",
                "- Fashion-MNIST quality is computed from the exported FP32 weights over the "
                "official 10,000-image test split. One exported sample is also checked through "
                "the C++ runtime.",
                "- Qwen correctness compares prefill and cached-decode logits against the "
                "stored reference fixture and requires identical greedy tokens.",
                "- Component benchmarks use their built-in warm-up and repeat policies. "
                "Memory-plan values are medians of five independent benchmark invocations. "
                "Paged-cache memory figures come from cache state rather than process RSS.",
                "- Results are local measurements, not cross-machine claims.",
                "",
                "## Figures",
                "",
                "- `figures/fashion_confusion_matrix.{png,svg}`",
                "- `figures/fashion_class_f1.{png,svg}`",
                "- `figures/qwen_logit_error.{png,svg}`",
                "- `figures/qwen_layer_max_error.{png,svg}`",
                "- `figures/qwen_layer_mean_error.{png,svg}`",
                "- `figures/qwen_generation_latency.{png,svg}`",
                "- `figures/kv_cache_speedup.{png,svg}`",
                "- `figures/paged_kv_memory.{png,svg}`",
                "- `figures/linear_threading_speedup.{png,svg}`",
                "- `figures/memory_plan_latency.{png,svg}`",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    if args.qwen_runs <= 0 or args.max_new_tokens <= 1:
        raise ValueError("qwen-runs must be positive and max-new-tokens must exceed one")

    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    copy_dirs = [path.resolve() for path in args.copy_dir]
    build_dir = args.build_dir.resolve()

    fashion = eval_fashion(
        args.fashion_model.resolve(),
        args.fashion_data.resolve(),
        build_dir / "tinyinfer",
    )
    qwen = eval_qwen(
        args.qwen_model.resolve(),
        build_dir / "check_qwen_model",
        build_dir / "tinyinfer",
        args.qwen_runs,
        args.max_new_tokens,
    )
    benchmarks = eval_benchmarks(build_dir)
    result = {
        "schema_version": 2,
        "environment": {
            "commit": run("git", "-C", ROOT, "rev-parse", "HEAD").strip(),
            "cpu": cpu_name(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "fashion_mnist": fashion,
        "qwen": qwen,
        "benchmarks": benchmarks,
    }

    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_csv(output_dir / "fashion_per_class.csv", fashion["per_class"])
    write_csv(output_dir / "fashion_confusion.csv", fashion["confusion"])
    write_csv(output_dir / "qwen_correctness.csv", qwen["correctness"])
    write_csv(output_dir / "qwen_layer_errors.csv", qwen["layer_metrics"])
    write_csv(output_dir / "qwen_generation.csv", qwen["performance"])
    write_csv(output_dir / "kv_cache.csv", benchmarks["kv_cache"])
    write_csv(output_dir / "paged_kv_cache.csv", benchmarks["paged_kv_cache"])
    write_csv(output_dir / "linear_threading.csv", benchmarks["linear"])
    write_csv(output_dir / "memory_plan_runs.csv", benchmarks["memory_plan_runs"])

    style()
    plot_fashion(fashion, figure_dir, copy_dirs)
    plot_qwen(qwen, figure_dir, copy_dirs)
    plot_benchmarks(benchmarks, figure_dir, copy_dirs)
    write_report(output_dir / "README.md", result)

    print(f"Fashion-MNIST accuracy: {fashion['accuracy']:.4f}")
    print(
        "Qwen maximum logit error: "
        f"{max(row['max_abs_error'] for row in qwen['correctness']):.9g}"
    )
    print(f"Wrote evaluation artifacts to {output_dir}")
    for directory in copy_dirs:
        print(f"Copied figures to {directory}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and export a small Fashion-MNIST MLP for tinyinfer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
import urllib.request
from pathlib import Path

import numpy as np


BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion"
)
FILES = {
    "train-images": (
        "train-images-idx3-ubyte.gz",
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    ),
    "train-labels": (
        "train-labels-idx1-ubyte.gz",
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    ),
    "test-images": (
        "t10k-images-idx3-ubyte.gz",
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    ),
    "test-labels": (
        "t10k-labels-idx1-ubyte.gz",
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    ),
}
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
EPOCHS = 15
HIDDEN_SIZE = 64
BATCH_SIZE = 256
LEARNING_RATE = 0.1
SEED = 0
SAMPLE_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("build/fashion-mnist-data"),
    )
    parser.add_argument(
        "--tinyinfer",
        type=Path,
        help="optional tinyinfer executable used to verify exported logits",
    )
    return parser.parse_args()


def download_dataset(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name, (filename, expected_hash) in FILES.items():
        path = directory / filename
        if not path.exists():
            print(f"downloading {filename}")
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {path}")
        paths[name] = path

    return paths


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        data = stream.read()

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(-1, 784).astype(np.float32) / 255.0


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        data = stream.read()

    return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)


def train(
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    input_size = images.shape[1]
    weight1 = (
        rng.standard_normal((input_size, HIDDEN_SIZE)).astype(np.float32)
        * np.float32(np.sqrt(2.0 / input_size))
    )
    bias1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    weight2 = (
        rng.standard_normal((HIDDEN_SIZE, len(CLASS_NAMES))).astype(np.float32)
        * np.float32(np.sqrt(2.0 / HIDDEN_SIZE))
    )
    bias2 = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    for epoch in range(EPOCHS):
        order = rng.permutation(images.shape[0])
        for begin in range(0, order.size, BATCH_SIZE):
            indices = order[begin : begin + BATCH_SIZE]
            inputs = images[indices]
            targets = labels[indices]

            preactivation = inputs @ weight1 + bias1
            hidden = np.maximum(preactivation, 0.0)
            logits = hidden @ weight2 + bias2
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            probabilities[np.arange(indices.size), targets] -= 1.0
            probabilities /= indices.size

            hidden_gradient = probabilities @ weight2.T
            hidden_gradient[preactivation <= 0.0] = 0.0
            weight2 -= LEARNING_RATE * (hidden.T @ probabilities)
            bias2 -= LEARNING_RATE * probabilities.sum(axis=0)
            weight1 -= LEARNING_RATE * (inputs.T @ hidden_gradient)
            bias1 -= LEARNING_RATE * hidden_gradient.sum(axis=0)

        print(f"epoch={epoch + 1}")

    return weight1, bias1, weight2, bias2


def write_model(
    directory: Path,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "model.ti"
    weights_path = directory / "weights.bin"
    arrays = (
        ("fc1.weight", weights[0]),
        ("fc1.bias", weights[1]),
        ("fc2.weight", weights[2]),
        ("fc2.bias", weights[3]),
    )

    entries: list[tuple[str, tuple[int, ...], int, int]] = []
    offset = 0
    with weights_path.open("wb") as stream:
        for name, array in arrays:
            payload = np.asarray(array, dtype="<f4").tobytes(order="C")
            stream.write(payload)
            entries.append((name, array.shape, offset, len(payload)))
            offset += len(payload)

    lines = [
        "TINYINFER_MODEL 1",
        "weights weights.bin",
        "",
        f"input input 2 1 {weights[0].shape[0]}",
        "output logits",
        "",
    ]
    for name, shape, tensor_offset, byte_size in entries:
        dimensions = " ".join(str(value) for value in shape)
        lines.append(
            f"tensor {name} f32 {len(shape)} {dimensions} "
            f"{tensor_offset} {byte_size}"
        )
    lines.extend(
        (
            "",
            "node fc1 Linear 3 input fc1.weight fc1.bias hidden",
            "node relu ReLU 1 hidden activated",
            "node fc2 Linear 3 activated fc2.weight fc2.bias logits",
            "",
            "end",
        )
    )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest_path


def verify_runtime(
    executable: Path,
    manifest: Path,
    input_path: Path,
    expected: np.ndarray,
) -> None:
    result = subprocess.run(
        [str(executable), "run", str(manifest), str(input_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output_line = next(
        line for line in result.stdout.splitlines() if line.startswith("output_values=")
    )
    actual = np.fromstring(output_line.partition("=")[2], sep=" ", dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"runtime output shape {actual.shape} != {expected.shape}")

    maximum_error = float(np.max(np.abs(actual - expected)))
    if maximum_error > 1e-4:
        raise ValueError(f"runtime logits differ by {maximum_error}")
    print(f"runtime_max_error={maximum_error:.9g}")


def main() -> None:
    args = parse_args()
    output_paths = (
        args.output_dir / "model.ti",
        args.output_dir / "weights.bin",
        args.output_dir / "sample.f32",
        args.output_dir / "expected.f32",
    )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, existing)))

    paths = download_dataset(args.data_dir)
    train_images = read_images(paths["train-images"])
    train_labels = read_labels(paths["train-labels"])
    test_images = read_images(paths["test-images"])
    test_labels = read_labels(paths["test-labels"])

    weights = train(train_images, train_labels)
    weight1, bias1, weight2, bias2 = weights
    test_logits = np.maximum(test_images @ weight1 + bias1, 0.0) @ weight2 + bias2
    predictions = np.argmax(test_logits, axis=1)
    accuracy = float(np.mean(predictions == test_labels))

    manifest = write_model(args.output_dir, weights)
    sample = test_images[SAMPLE_INDEX : SAMPLE_INDEX + 1]
    expected = test_logits[SAMPLE_INDEX]
    input_path = args.output_dir / "sample.f32"
    expected_path = args.output_dir / "expected.f32"
    input_path.write_bytes(np.asarray(sample, dtype="<f4").tobytes(order="C"))
    expected_path.write_bytes(np.asarray(expected, dtype="<f4").tobytes(order="C"))

    label = int(test_labels[SAMPLE_INDEX])
    prediction = int(predictions[SAMPLE_INDEX])
    print(f"test_accuracy={accuracy:.4f}")
    print(f"sample_label={label} ({CLASS_NAMES[label]})")
    print(f"sample_prediction={prediction} ({CLASS_NAMES[prediction]})")
    print(f"wrote {manifest}")
    print(f"wrote {input_path}")
    print(f"wrote {expected_path}")

    if args.tinyinfer:
        verify_runtime(args.tinyinfer.resolve(), manifest.resolve(), input_path.resolve(), expected)


if __name__ == "__main__":
    main()

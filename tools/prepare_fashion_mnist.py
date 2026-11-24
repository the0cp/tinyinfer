#!/usr/bin/env python3
"""Train and export a small Fashion-MNIST MLP for tinyinfer."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST


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


def load_dataset(directory: Path) -> tuple[torch.Tensor, ...]:
    train = FashionMNIST(directory, train=True, download=True)
    test = FashionMNIST(directory, train=False, download=True)

    train_images = train.data.reshape(-1, 784).float() / 255.0
    test_images = test.data.reshape(-1, 784).float() / 255.0
    return train_images, train.targets, test_images, test.targets


def train_model(images: torch.Tensor, labels: torch.Tensor) -> nn.Sequential:
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(784, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, len(CLASS_NAMES)),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
    )
    loss_function = nn.CrossEntropyLoss()
    batches = DataLoader(
        TensorDataset(images, labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model.train()
    for epoch in range(EPOCHS):
        for inputs, targets in batches:
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1}")

    return model.eval()


def write_model(directory: Path, model: nn.Sequential) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "model.ti"
    weights_path = directory / "weights.bin"
    arrays = (
        ("fc1.weight", model[0].weight.T),
        ("fc1.bias", model[0].bias),
        ("fc2.weight", model[2].weight.T),
        ("fc2.bias", model[2].bias),
    )

    entries: list[tuple[str, tuple[int, ...], int, int]] = []
    offset = 0
    with weights_path.open("wb") as stream:
        for name, tensor in arrays:
            array = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
            payload = array.tobytes(order="C")
            stream.write(payload)
            entries.append((name, array.shape, offset, len(payload)))
            offset += len(payload)

    lines = [
        "TINYINFER_MODEL 1",
        "weights weights.bin",
        "",
        "input input 2 1 784",
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

    train_images, train_labels, test_images, test_labels = load_dataset(args.data_dir)
    model = train_model(train_images, train_labels)
    with torch.inference_mode():
        test_logits = model(test_images)
        predictions = test_logits.argmax(dim=1)

    accuracy = float((predictions == test_labels).float().mean())
    manifest = write_model(args.output_dir, model)
    sample = test_images[SAMPLE_INDEX : SAMPLE_INDEX + 1].numpy().astype("<f4")
    expected = test_logits[SAMPLE_INDEX].numpy().astype("<f4")
    input_path = args.output_dir / "sample.f32"
    expected_path = args.output_dir / "expected.f32"
    input_path.write_bytes(sample.tobytes(order="C"))
    expected_path.write_bytes(expected.tobytes(order="C"))

    label = int(test_labels[SAMPLE_INDEX])
    prediction = int(predictions[SAMPLE_INDEX])
    print(f"test_accuracy={accuracy:.4f}")
    print(f"sample_label={label} ({CLASS_NAMES[label]})")
    print(f"sample_prediction={prediction} ({CLASS_NAMES[prediction]})")
    print(f"wrote {manifest}")
    print(f"wrote {input_path}")
    print(f"wrote {expected_path}")

    if args.tinyinfer:
        verify_runtime(
            args.tinyinfer.resolve(),
            manifest.resolve(),
            input_path.resolve(),
            expected,
        )


if __name__ == "__main__":
    main()

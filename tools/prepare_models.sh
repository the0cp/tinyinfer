#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if command -v uv >/dev/null 2>&1; then
    uv_command="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_command="$HOME/.local/bin/uv"
else
    echo "uv is required; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

environment=build/model-env
python="$environment/bin/python"
if [[ ! -x "$python" ]]; then
    "$uv_command" venv --python 3.12 "$environment"
fi

"$uv_command" pip install --python "$python" \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu
"$uv_command" pip install --python "$python" \
    -r tools/qwen-reference-requirements.txt \
    -r tools/evaluation-requirements.txt

qwen_source=models/sources/qwen2.5-0.5b-instruct-hf
qwen_output=models/qwen2.5-0.5b
if [[ ! -f "$qwen_source/model.safetensors" ]]; then
    "$environment/bin/huggingface-cli" download Qwen/Qwen2.5-0.5B-Instruct \
        --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
        --local-dir "$qwen_source"
fi

if [[ ! -f "$qwen_output/model.tiq" ]]; then
    "$python" tools/convert_qwen2.py \
        --model-dir "$qwen_source" \
        --output-dir "$qwen_output"
fi

if [[ ! -f "$qwen_output/model_case.bin" || ! -f "$qwen_output/fixtures.json" ]]; then
    "$python" tools/export_qwen_fixtures.py \
        --model-dir "$qwen_source" \
        --output-dir "$qwen_output"
fi

if [[ ! -f models/fashion-mnist-final/model.ti ]]; then
    "$python" tools/prepare_fashion_mnist.py \
        --data-dir models/fashion-mnist-data \
        --output-dir models/fashion-mnist-final \
        --tinyinfer build-release/tinyinfer
fi

echo "models are ready under $repo_root/models"

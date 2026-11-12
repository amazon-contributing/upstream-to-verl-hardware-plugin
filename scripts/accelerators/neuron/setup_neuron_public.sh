#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
environment_root=${1:?Usage: setup_neuron_public.sh /absolute/environment/root}
[[ "$environment_root" == /* ]] || { echo "Use an absolute environment root." >&2; exit 1; }
: "${NEURON_TRAINING_REQUIREMENTS:?Set the absolute path to your training requirements file}"
: "${NEURON_SERVING_REQUIREMENTS:?Set the absolute path to your serving requirements file}"
for requirements_file in "$NEURON_TRAINING_REQUIREMENTS" "$NEURON_SERVING_REQUIREMENTS"; do
    [[ "$requirements_file" == /* && -f "$requirements_file" && -r "$requirements_file" ]] || {
        echo "Requirements must be an existing readable absolute file: $requirements_file" >&2; exit 1;
    }
done
uv_bin=${UV_BIN:-uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-"$environment_root/python"}
export UV_PYTHON_BIN_DIR=${UV_PYTHON_BIN_DIR:-"$environment_root/bin"}
mkdir -p "$environment_root/sources"
"$uv_bin" python install 3.12.13
python_bin=$("$uv_bin" python find --managed-python 3.12.13)

checkout_source() {
    local name=$1 url=$2 revision=$3 directory="$environment_root/sources/$1"
    if [[ ! -d "$directory/.git" ]]; then
        git init -q "$directory"
        git -C "$directory" remote add origin "$url"
        git -C "$directory" fetch -q --depth 1 origin "$revision"
        git -C "$directory" checkout -q --detach FETCH_HEAD
    fi
    [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || {
        echo "$name: existing checkout does not match $revision" >&2; exit 1;
    }
    git -C "$directory" diff --quiet HEAD || { echo "$name has local changes" >&2; exit 1; }
}

checkout_source torchtitan-neuron https://github.com/aws-neuron/torchtitan-neuron.git b7582e986586051d7be85d03956afe169624aeda
checkout_source vllm-neuron https://github.com/vllm-project/vllm-neuron.git eb6cb360f7cff6082e22a9437d8ce0ba3891304a

for role in training serving; do
    environment="$environment_root/$role"
    if [[ ! -x "$environment/bin/python" ]]; then
        "$uv_bin" venv --python "$python_bin" "$environment"
    fi
    requirements_file=$NEURON_TRAINING_REQUIREMENTS
    [[ "$role" == serving ]] && requirements_file=$NEURON_SERVING_REQUIREMENTS
    # Use complete, externally maintained requirements without resolving
    # CUDA/audio extras over the pinned CPU PyTorch.
    "$uv_bin" pip install --python "$environment/bin/python" --no-deps \
        --extra-index-url https://pip.repos.neuron.amazonaws.com \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match -r "$requirements_file"
    source_name=torchtitan-neuron
    [[ "$role" == serving ]] && source_name=vllm-neuron
    "$uv_bin" pip install --python "$environment/bin/python" --no-deps --no-build-isolation \
        -e "$environment_root/sources/$source_name" -e "$ROOT"
    "$environment/bin/python" - <<'PY'
import sys
import sysconfig
from pathlib import Path
prefix = Path(sys.prefix)
runtime = prefix / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
links = prefix / "lib/nixl-cuda-stubs"
links.mkdir(exist_ok=True)
link = links / "libcudart.so.12"
if not link.exists():
    link.symlink_to(runtime)
# TorchXLA, imported by the public lite runtime, links against libpython.
shared_python = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
if shared_python.exists():
    for name in {shared_python.name, sysconfig.get_config_var("INSTSONAME")} - {None}:
        python_link = prefix / "lib" / name
        if not python_link.exists():
            python_link.symlink_to(shared_python)
PY
done

export VERL_NEURON_TRAIN_PYTHON="$environment_root/training/bin/python"
export VERL_NEURON_ROLLOUT_PYTHON="$environment_root/serving/bin/python"
"$VERL_NEURON_ROLLOUT_PYTHON" "$ROOT/scripts/accelerators/neuron/check_neuron_environment.py"
printf 'export VERL_NEURON_TRAIN_PYTHON=%q\nexport VERL_NEURON_ROLLOUT_PYTHON=%q\n' \
    "$VERL_NEURON_TRAIN_PYTHON" "$VERL_NEURON_ROLLOUT_PYTHON"

# Neuron validation on verl 0.9.1

Run commands from this checkout inside the pinned public container, with the
two interpreter variables from the [guide](README.md). Hardware tests require
idle allocated cores and the documented host driver.

## CPU and cross-environment checks

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VERL_PLATFORM=nvidia VLLM_PLUGINS=''
export USE_TORCH_XLA=0 VERL_USE_EXTERNAL_PLUGINS=none OMP_NUM_THREADS=1
"$VERL_NEURON_TRAIN_PYTHON" -m pytest -q \
  tests/accelerators/neuron tests/test_registration_dispatch.py
"$VERL_NEURON_ROLLOUT_PYTHON" -m pytest -q tests/accelerators/neuron
RUN_NEURON_PUBLIC_RAY_TESTS=1 "$VERL_NEURON_ROLLOUT_PYTHON" -m pytest \
  tests/accelerators/neuron/test_public_runtime.py -k ray -sx
```

SDK-dependent tests skip on ordinary CPU CI. Execute the real TorchTitan
statistics/gradient tests in training and the real vLLM sampler tests in serving.

## Hardware gates

```bash
VERL_PLATFORM=aws NEURON_RT_VISIBLE_CORES=0-1 NEURON_LOGICAL_NC_CONFIG=2 \
  RUN_NEURON_HARDWARE_TESTS=1 "$VERL_NEURON_TRAIN_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m pytest \
  tests/accelerators/neuron/test_tensor_parallel.py -k native -sx

VERL_PLATFORM=aws NEURON_RT_VISIBLE_CORES=0-1 NEURON_LOGICAL_NC_CONFIG=2 \
  RUN_NEURON_HARDWARE_TESTS=1 MODEL_PATH=/models/qwen3-0.6b-untied \
  NEURON_TEST_CHECKPOINT_DIR=/runs/new-engine-checkpoint \
  "$VERL_NEURON_TRAIN_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m pytest tests/accelerators/neuron/test_native_hardware.py -sx

RUN_NEURON_NIXL_TESTS=1 TRAINER_CORE=4 ROLLOUT_CORE=8 \
  "$VERL_NEURON_ROLLOUT_PYTHON" -m pytest tests/accelerators/neuron/test_nixl_hardware.py -sx

VERL_PLATFORM=aws VERL_NEURON_SERVING=1 VLLM_PLUGINS=neuron \
  NEURON_RT_VISIBLE_CORES=2 NEURON_LOGICAL_NC_CONFIG=2 NEURON_TEST_SAMPLER_DEVICE=neuron:0 \
  "$VERL_NEURON_ROLLOUT_PYTHON" -m pytest \
  tests/accelerators/neuron/test_worker_extension.py -k real_vllm_sampler -sx

VERL_PLATFORM=aws VERL_NEURON_SERVING=1 NEURON_RT_VISIBLE_CORES=2 \
  NEURON_LOGICAL_NC_CONFIG=2 NEURON_TEST_LAYOUT_DEVICE=neuron:0 \
  "$VERL_NEURON_ROLLOUT_PYTHON" -m pytest \
  tests/accelerators/neuron/test_weight_layout.py -k dense_tp_boxes -sx
```

Repeat the engine gate with eight cores, TP4/FSDP2 and full activation
checkpointing (`NEURON_TEST_TP=4 NEURON_TEST_AC=full`). Then use 32 cores and the
4B model with `NEURON_TEST_TP=4 NEURON_TEST_AC=full NEURON_TEST_REFERENCE=1
NEURON_TEST_SEQ_LEN=2048 NEURON_TEST_MICRO_BATCH=4`. Use a fresh checkpoint
directory for each gate.

## End-to-end acceptance

Use an external two-step configuration with checkpoint saving enabled. Require finite nonzero
gradients, changing checkpoint and receiver weights, three policy
synchronizations, two checkpoints, and checkpoint recovery.

For a full 4B run, require every policy synchronization,
finite scalars and nonzero updates, all scheduled/final GSM8K evaluations, and
final accuracy above initial accuracy. Retain all 1,319 validation examples and
generations per evaluation. Record resolved configs, versions, logs, checkpoints
from recovery gates, and learning curves. Historical curves provide context;
numerical or performance equivalence is not an acceptance claim.

## Port validation status

Software validation completed on 2026-11-10 using fresh training and serving
environments installed by `setup_neuron_public.sh`. Both contain the published
`verl==0.9.1` wheel. The source port is `7bf75e2`; the development branch is
`feat/neuron-torchtitan-vllm`.

| Check | Result |
| --- | --- |
| Neuron and registration tests, training interpreter | 164 passed, 28 skipped |
| Neuron suite, serving interpreter | 137 passed, 7 skipped |
| Ray serving-controller → training-worker batch exchange | 1 passed |
| Neuron-only static API check against the 0.9.1 release | 26 imports checked, 0 missing |
| Ruff lint and formatting | Passed |
| Mypy with isolated CI tooling | Passed, 130 source files |
| Pre-commit hooks on changed files | Passed |
| License/copyright checks and shell syntax | Passed |
| Wheel build and packaged launcher inspection | Passed |
| Native trainer → lite receiver NIXL/LIBFABRIC | Passed: 10 writes, 41,943,188 bytes, suffix checks |
| Qwen3-4B GRPO on 64 logical NeuronCores | Completed 56 steps, 57 policy synchronizations, 13 evaluations |

The serving suite exercises the installed vLLM sampler. The launcher
regression verifies that vLLM discovery messages cannot enter Hydra arguments and
that both interpreter paths and runtime environment settings survive config
composition. Skipped tests include hardware-only and unavailable optional backend
cases. A mypy run inside the full SDK environment exposed eight existing XPU
`pyzmq` attribute errors; the repository's isolated CI-style invocation passes.
Running the pinned pre-commit hooks over the entire repository also reports
existing MLU lint and TPU/test-stub typing issues outside this change.
The unmodified repository-wide API checker reports nine entries in existing
MLU/TPU code. The Neuron-only scan passes with that same checker.

Local logs, the environment manifest, API report and wheel are retained in
`/home/ubuntu/.cache/verl-neuron-stable-091/`. The managed worktree is
`/home/ubuntu/.codex/worktrees/cbee/upstream-to-verl-hardware-plugin`.

## Completed Qwen3-4B run

The 56-step run completed on its first attempt with exit code 0, from
2026-11-10 07:54 to 13:18 UTC (about 5 hours 24 minutes including startup).
Progress was checked every 30 minutes. No training failure or relaunch occurred.
The validation configuration used 32 trainer cores (TP4/FSDP8) and four TP8
rollout replicas on the pinned public environments with `verl==0.9.1`.
It used 512 prompts × five samples, micro-batch four, 1024+1024 tokens,
LR `1e-6`, KL `0.001`, full actor activation checkpointing, two optimizer
updates per synchronization, and no prefetch. The exact resolved configuration
and launch-time source patch are retained with the run artifacts. Workload
recipes were subsequently removed from the port; the backend implementation
used for this run is unchanged.

Final GSM8K accuracy was **92.95% (1,226/1,319)**, compared with **39.27%
(518/1,319)** initially. The highest recorded accuracy was 93.56% at step 50.
Every evaluation retained the same 1,319 unique input/answer pairs and nonempty
generations. Rescoring every saved generation with the installed release's GSM8K
reward function reproduced every saved score and logged accuracy.

| Step | Correct / 1,319 | Accuracy |
| --- | ---: | ---: |
| 0 | 518 | 39.27% |
| 5 | 880 | 66.72% |
| 10 | 1,016 | 77.03% |
| 15 | 1,124 | 85.22% |
| 20 | 1,160 | 87.95% |
| 25 | 1,191 | 90.30% |
| 30 | 1,207 | 91.51% |
| 35 | 1,220 | 92.49% |
| 40 | 1,223 | 92.72% |
| 45 | 1,233 | 93.48% |
| 50 | 1,234 | 93.56% |
| 55 | 1,230 | 93.25% |
| 56 | 1,226 | 92.95% |

All recorded scalars were finite. Every training step had positive gradients
(norm range 0.09045–0.18510). All 57 policy synchronizations, including
initialization, delivered their final bucket to all 32 rollout workers.
Mean logged step time over steps 2–56 was 310.56 seconds; mean NIXL send time was
3.93 seconds. Validation time is separate.

Artifacts are under `runs/full-20261110T075406Z/` in the local artifact directory
above: `manifest.json`, `implementation.patch`, `environment.json`,
`resolved-config.json`, `train.log`, `progress.jsonl`, `checks.jsonl`,
`training-metrics.csv`, `evaluations.csv`, `acceptance-audit.json`, and all
`validation/*.jsonl` generations.

A DataLoader-worker traceback appeared in `weakref._exitfunc` during interpreter
teardown, after final validation was recorded. The process returned 0 and no
training processes remained afterward. The audit retains this traceback and
separately verifies all completed steps, transfers and evaluations; it does not
classify the teardown message as a training failure.

The host temporarily used Neuron driver `2.31.7.0`. After validation, the owned
container was stopped and driver `2.30.2.0` and all saved parameters were restored.
All 16 devices were idle after cleanup.

This validation run disabled checkpoint saving. Standalone native-engine
and two-step checkpoint-recovery gates have not been rerun for the stable port.

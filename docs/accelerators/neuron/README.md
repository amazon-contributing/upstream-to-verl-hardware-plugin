# AWS Trainium: TorchTitan and vLLM-Neuron

This experimental backend runs single-host GRPO with native Neuron TorchTitan
training and vLLM-Neuron rollout using the published **verl 0.9.1** release.
The plugin owns platform integration, dense HF-to-TP weight layouts, and direct
NIXL/LIBFABRIC transport.

## Supported configuration

- Dense Qwen3, BF16, actor and frozen reference, rule-based GRPO rewards.
- FSDP2 × tensor parallel training and independent TP-only rollout replicas.
- Fixed-width, right-padded causal SDPA inputs with sequence lengths divisible
  by 512. Differentiable token statistics move to the host for verl's loss;
  model execution and optimization stay on Trainium.
- Vocabulary-parallel log probabilities delegate to TorchTitan's unreduced loss.
  Entropy supports gradients without gathering vocabulary logits.
- Synchronous model/optimizer/scheduler checkpoints on a shared filesystem.

Training attention heads, KV heads, vocabulary and FFN width must divide by TP.
CP, EP and PP must be one. Packed inputs, model/optimizer offloading, MoE,
quantization, learned critics, multimodal inputs, colocated rollout and multi-node
jobs are outside the supported configuration.

Registry keys are `VERL_PLATFORM=aws`, device `neuron`, strategy `torchtitan`,
rollout `vllm`, and checkpoint backend `neuron_nixl`. Actor/reference configs use
`verl_hardware_plugin.accelerators.neuron.engines.config.NeuronTorchtitanEngineConfig`.

## Public dependency pins

| Component | Release / revision |
| --- | --- |
| verl | PyPI `0.9.1`; release commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` |
| TorchTitan Neuron | `release-2.33.0`, `b7582e986586051d7be85d03956afe169624aeda` |
| vLLM-Neuron | `v0.29.0.1.2.0`, `eb6cb360f7cff6082e22a9437d8ce0ba3891304a` |
| Training PyTorch / TorchNeuron | `2.14.0+cpu` / `2.14.3.0.281+dc70ffa` |
| Serving PyTorch / lite | `2.13.0+cpu` / `2.13.0.1.0.3112+653bc48f` |
| vLLM / Ray / TransferQueue | `0.29.0` / `2.55.0` / `0.1.11` |

Both environments use managed Python 3.12.13. Complete pins are in
[training requirements](../../../requirements/neuron-training.txt) and
[serving requirements](../../../requirements/neuron-serving.txt).
Installation uses `--no-deps` to preserve CPU PyTorch wheels.
Controllers and reward workers use the serving interpreter without importing
TorchNeuron. A packaged shell launcher selects the interpreter before Python
starts and preserves Ray's assigned NeuronCores.

The [container](../../../docker/Dockerfile.neuron-public) supplies Ubuntu 26.04
(the pinned TorchNeuron binary needs GLIBC 2.43), Neuron runtime
`2.35.30.0-d2725ccb8`, collectives `2.35.43.0-354ad3064`, and EFA userspace
libfabric `2.7.0amzn2.0`. The host must supply Neuron driver **2.31.7.0** and EFA
kernel support. Setup does not install or reload host drivers.
NIXL uses CUDA 12 libraries/stubs even for Neuron HBM; do not install `nixl-cu13`
or replace the container's EFA libraries with older host libraries.

## Installation and launch

Build from the plugin checkout:

```bash
docker build -f docker/Dockerfile.neuron-public -t verl-neuron-public:2.33 .
```

Run on an idle Trainium2 host. Set absolute `PUBLIC_ENVS`, `MODEL_DIR` and
`DATA_DIR` paths before this command:

```bash
mkdir -p "$PUBLIC_ENVS"
neuron_devices=()
for device in /dev/neuron[0-9]*; do neuron_devices+=(--device "$device"); done
docker run --rm -it --network host --ipc host --ulimit memlock=-1:-1 \
  "${neuron_devices[@]}" --device /dev/infiniband \
  -v "$PWD:$PWD" -w "$PWD" \
  -v "$PUBLIC_ENVS:$PUBLIC_ENVS" \
  -v "$MODEL_DIR:$MODEL_DIR:ro" -v "$DATA_DIR:$DATA_DIR:ro" \
  -e PUBLIC_ENVS="$PUBLIC_ENVS" \
  -e UV_PYTHON_INSTALL_DIR="$PUBLIC_ENVS/python" \
  -e UV_CACHE_DIR="$PUBLIC_ENVS/uv-cache" \
  -e HF_HOME="$PUBLIC_ENVS/hf-cache" \
  -e VLLM_CACHE_ROOT="$PUBLIC_ENVS/vllm-cache" \
  -e NKI_COMPILE_CACHE_URL="$PUBLIC_ENVS/nki-cache" \
  verl-neuron-public:2.33 bash
```

Inside the container:

```bash
bash scripts/accelerators/neuron/setup_neuron_public.sh "$PUBLIC_ENVS"
export VERL_NEURON_TRAIN_PYTHON="$PUBLIC_ENVS/training/bin/python"
export VERL_NEURON_ROLLOUT_PYTHON="$PUBLIC_ENVS/serving/bin/python"
"$VERL_NEURON_ROLLOUT_PYTHON" scripts/accelerators/neuron/check_neuron_environment.py --hardware

CONFIG_PATH=/absolute/path/to/your/configs CONFIG_NAME=your_training_config \
  RUN_DIR=/runs/neuron \
  bash scripts/accelerators/neuron/run_neuron.sh
```

The launcher accepts an external Hydra configuration and additional overrides;
workload examples and dataset recipes are not bundled. Use the mounted model and
data paths in your configuration. Compose verl's `ppo_trainer` with
`model_engine: torchtitan` and `pkg://verl.trainer.config` in Hydra's search path.
Apply the following backend settings:

| Configuration | Required value |
| --- | --- |
| `trainer.device`, `trainer.use_v1` | `neuron`, `true` |
| `trainer.v1.trainer_mode`, `actor_rollout_ref.hybrid_engine` | `separate_async`, `false` |
| Actor/reference `torchtitan._target_` | `verl_hardware_plugin.accelerators.neuron.engines.config.NeuronTorchtitanEngineConfig` |
| Actor/reference `torchtitan.attn_type`, `spmd_backend` | `sdpa`, `default` |
| Model and actor/reference `torchtitan.use_remove_padding` | `false` |
| Model `use_fused_kernels` | `false` |
| Actor `use_dynamic_bsz`; reference/rollout `log_prob_use_dynamic_bsz` | `false` |
| Rollout `name`, `dtype`, `load_format` | `vllm`, `bfloat16`, `auto` |
| Rollout `enable_sleep_mode`, `free_cache_engine`, `enable_prefix_caching`, `enable_chunked_prefill` | `false` |
| Rollout `calculate_log_probs` | `true` |
| Rollout `checkpoint_engine.backend` | `neuron_nixl` |
| Rollout `checkpoint_engine.custom_backend_module` | `verl_hardware_plugin.accelerators.neuron.weight_sync.checkpoint_engine` |

Set actor/reference `torchtitan.max_seq_len` to the prompt plus response length,
divisible by 512, and set rollout length/token limits accordingly. Disable model
and optimizer offloading. Set fixed micro-batch sizes and parallel dimensions for
your allocation. `trainer.n_gpus_per_node` counts training NeuronCores; rollout
`n_gpus_per_node` counts separate rollout NeuronCores. Both use one node.

The launcher refuses an existing Ray cluster or occupied NeuronCores. Cold native
compilation can take more than 15 minutes. Preserve NKI
binaries together with the vLLM cache: compiled graphs contain absolute kernel paths.

## Stable verl compatibility and synchronization

The plugin adapts 0.9.1 without editing the installed release. Ray checks count
`neuron_cores` and allocations remain exclusive. The launcher explicitly forwards
runtime settings because 0.9.1 does not consume the platform's Ray hook.
The Neuron server returns collective-RPC acknowledgements, validates effective
sampling settings, and checks sampled-token log probabilities. A trainer-local
logging adapter handles the release's `flush=True` log call.

Native sampling supports temperature, top-k and top-p. Scoring follows the
device-selected token and native sampler's candidate cap (normally 256) and
top-p convention. Unsupported penalties, seeded sampling, custom processors,
speculative decoding and jump-forward decoding fail explicitly. Optional CPU
sampling keeps metadata on the host and avoids CUDA/Triton dispatch.

All trainer ranks export deterministic HF tensors. Senders distribute receiver
TP layouts round-robin and transfer packed BF16 payloads over NIXL. TCPStore
carries metadata and acknowledgements only. Receivers validate full weight
coverage and apply each bucket before acknowledging slot reuse. Failed updates
keep generation paused and do not advance the policy version.

The lite receiver stages buckets through host memory when installing weights.
Transport remains HBM-to-HBM NIXL/LIBFABRIC. Export casts local shards to BF16
before gathering. `weight_sync_cpu_gather_threshold_bytes` defaults to zero; a
positive value selects host gathering for large exports. Bucket size bounds
transport buffers, not the largest gathered parameter.

See [validation](validation.md) for commands and current results. This port derives
from source plugin commit `7bf75e2`.
[Historical public-stack results](https://github.com/czkkkkkk/upstream-verl-hardware-plugin/blob/7bf75e2/docs/accelerators/neuron/public_stack.md)
used a development verl revision and completed only 26 of 56 steps. They are
comparison evidence, not validation of this stable-release port.

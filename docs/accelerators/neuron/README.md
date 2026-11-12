# Run verl on AWS Trainium

This experimental backend runs **verl 0.9.1** with native Neuron TorchTitan
training and vLLM-Neuron rollout. This guide covers environment setup and a small
GRPO example using Qwen3-0.6B and GSM8K on one Trainium2 host.

## 1. Set up the environment

### Prepare the host

Use an idle Trainium2 host with access to its Neuron and EFA devices. The
validated system environment is:

| Component | Version / requirement |
| --- | --- |
| OS | Ubuntu 26.04; the pinned TorchNeuron binary requires GLIBC 2.43 |
| Neuron driver | `2.31.7.0` |
| Neuron runtime | `2.35.30.0-d2725ccb8` |
| Neuron collectives | `2.35.43.0-354ad3064` |
| EFA userspace libfabric | `2.7.0amzn2.0`, with EFA kernel support |
| System libraries | `libnuma-dev` for `libnuma.so`; CUDA 12 libraries/stubs, including `libcuda.so.1`, for NIXL |
| Setup tools | Git, curl, and [uv](https://docs.astral.sh/uv/getting-started/installation/) `0.11.14` |

Provision these system dependencies before running the installer. The installer
creates Python environments; it does not install host drivers, EFA, or system
libraries.

### Install the Python dependencies

Training and serving require different PyTorch and TorchNeuron versions, so the
[installer](../../../scripts/accelerators/neuron/setup_neuron_public.sh) creates
two environments:

| Environment | Purpose | Main dependencies |
| --- | --- | --- |
| `training` | Actor and frozen reference | PyTorch `2.14.0+cpu`, TorchNeuron `2.14.3.0.281+dc70ffa`, TorchTitan Neuron `release-2.33.0` |
| `serving` | Rollout, controller, and reward workers | PyTorch `2.13.0+cpu`, lite `2.13.0.1.0.3112+653bc48f`, vLLM `0.29.0`, vLLM-Neuron `v0.29.0.1.2.0` |

Both use managed Python 3.12.13, verl 0.9.1, and this plugin installed in editable
mode. The installer pins the required runtime packages and source revisions,
then uses uv to resolve transitive dependencies. No external requirements files
are needed. It excludes unused CUDA kernels, audio/video packages, and
`nixl-cu13`; generic dependency checks may report these intentional exclusions.

Run the following from the plugin checkout. Set `PUBLIC_ENVS` to an absolute,
writable path with room for both environments, model weights, and compiler caches.
Keep using the same shell for the example below.

```bash
export PLUGIN_DIR="$PWD"
export PUBLIC_ENVS=/absolute/path/to/neuron-envs
export LD_LIBRARY_PATH="/opt/aws/neuron/lib:/opt/amazon/efa/lib:/opt/amazon/efa/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash "$PLUGIN_DIR/scripts/accelerators/neuron/setup_neuron_public.sh" "$PUBLIC_ENVS"

export VERL_NEURON_TRAIN_PYTHON="$PUBLIC_ENVS/training/bin/python"
export VERL_NEURON_ROLLOUT_PYTHON="$PUBLIC_ENVS/serving/bin/python"
"$VERL_NEURON_ROLLOUT_PYTHON" \
  "$PLUGIN_DIR/scripts/accelerators/neuron/check_neuron_environment.py" --hardware
```

Resolve any environment-check failures before launching training. There is no
need to activate either environment: the launcher selects the appropriate
interpreter for each worker.

The example uses the default uv, Hugging Face, and vLLM cache locations.
The launcher defaults `NKI_COMPILE_CACHE_URL` to `$PUBLIC_ENVS/nki-cache`.
Set `UV_CACHE_DIR`, `HF_HOME`, `VLLM_CACHE_ROOT`, or `NKI_COMPILE_CACHE_URL` only
if you need a different cache location.

## 2. Run a minimal GRPO example

This example uses **four logical NeuronCores**: two for the actor and frozen
reference, and two for one rollout replica. Both use tensor parallelism of two.
On Trainium2, the launcher defaults to `NEURON_LOGICAL_NC_CONFIG=2`.
The verl fields named `n_gpus_per_node` count NeuronCores in this backend.

The run uses four prompts per step, two responses per prompt, and two training
steps. It is a smoke run for setup, generation, training, and weight
synchronization; it is too short to assess model quality.

### Prepare the model and dataset

Download Qwen3-0.6B and use the GSM8K preprocessing script from the pinned verl
release. The script creates the prompt and answer fields used by verl's built-in
GSM8K reward function.

```bash
export NEURON_EXAMPLE_DIR="$PUBLIC_ENVS/examples/gsm8k"
export MODEL_DIR="$NEURON_EXAMPLE_DIR/Qwen3-0.6B"
export DATA_DIR="$NEURON_EXAMPLE_DIR/data"
export CONFIG_PATH="$NEURON_EXAMPLE_DIR/config"
mkdir -p "$DATA_DIR" "$CONFIG_PATH"

# Use the serving environment for data preparation and the controller.
export VERL_PLATFORM=aws
export VERL_NEURON_PROCESS_ROLE=controller
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_PLUGINS=""
export LD_LIBRARY_PATH="$PUBLIC_ENVS/serving/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PUBLIC_ENVS/serving/bin/hf" download Qwen/Qwen3-0.6B --local-dir "$MODEL_DIR"
curl -fsSL \
  https://raw.githubusercontent.com/verl-project/verl/1876b06d0a3e4e71e06230be10af14492ca8a75b/examples/data_preprocess/gsm8k.py \
  -o "$NEURON_EXAMPLE_DIR/prepare_gsm8k.py"
"$VERL_NEURON_ROLLOUT_PYTHON" "$NEURON_EXAMPLE_DIR/prepare_gsm8k.py" \
  --local_save_dir "$DATA_DIR"
```

### Create the training configuration

This configuration extends verl's `ppo_trainer` defaults. The actor and reference
share the Neuron engine settings through a YAML anchor.

```bash
cat > "$CONFIG_PATH/neuron_gsm8k.yaml" <<'YAML'
defaults:
  - ppo_trainer
  - override model_engine: torchtitan
  - _self_

hydra:
  searchpath:
    - pkg://verl.trainer.config

data:
  train_files: ${oc.env:DATA_DIR}/train.parquet
  val_files: ${oc.env:DATA_DIR}/test.parquet
  train_max_samples: 32
  val_max_samples: 8
  train_batch_size: 4
  max_prompt_length: 512
  max_response_length: 512
  filter_overlong_prompts: true
  dataloader_num_workers: 0
  apply_chat_template_kwargs:
    enable_thinking: false

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  hybrid_engine: false
  model:
    path: ${oc.env:MODEL_DIR}
    use_remove_padding: false
    use_fused_kernels: false
  actor:
    ppo_mini_batch_size: 4
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: false
    use_kl_loss: true
    kl_loss_coef: 0.001
    optim:
      lr: 1.0e-6
    torchtitan: &neuron_engine
      _target_: verl_hardware_plugin.accelerators.neuron.engines.config.NeuronTorchtitanEngineConfig
      attn_type: sdpa
      spmd_backend: default
      dtype: bfloat16
      use_remove_padding: false
      activation_checkpoint: none
      max_seq_len: 1024
      tensor_parallel_size: 2
      data_parallel_size: 1
      data_parallel_replicate_size: 1
      data_parallel_shard_size: 1
      param_offload: false
      optimizer_offload: false
  ref:
    log_prob_micro_batch_size_per_gpu: 1
    log_prob_use_dynamic_bsz: false
    torchtitan:
      <<: *neuron_engine
      forward_only: true
  rollout:
    name: vllm
    nnodes: 1
    n_gpus_per_node: 2
    tensor_model_parallel_size: 2
    dtype: bfloat16
    load_format: auto
    n: 2
    temperature: 1.0
    top_k: 256
    top_p: 1.0
    max_model_len: 1024
    max_num_batched_tokens: 1024
    max_num_seqs: 4
    calculate_log_probs: true
    log_prob_micro_batch_size_per_gpu: 1
    log_prob_use_dynamic_bsz: false
    enable_sleep_mode: false
    free_cache_engine: false
    enable_prefix_caching: false
    enable_chunked_prefill: false
    agent:
      num_workers: 1
    checkpoint_engine:
      backend: neuron_nixl
      custom_backend_module: verl_hardware_plugin.accelerators.neuron.weight_sync.checkpoint_engine
      update_weights_bucket_megabytes: 64

reward:
  num_workers: 1

trainer:
  device: neuron
  use_v1: true
  nnodes: 1
  n_gpus_per_node: 2
  total_epochs: 1
  total_training_steps: 2
  project_name: neuron_gsm8k
  experiment_name: qwen3_0_6b_smoke
  logger: [console]
  val_before_train: false
  test_freq: -1
  save_freq: -1
  resume_mode: disable
  v1:
    trainer_mode: separate_async
    separate_async:
      parameter_sync_step: 1
YAML
```

The training sequence length is `512 + 512 = 1024`. Keep
`torchtitan.max_seq_len` equal to the prompt plus response limit and divisible by
512 when changing these values. This backend uses fixed micro-batches and padded
inputs. CP, EP, and PP remain one; packed inputs, offloading, MoE, quantization,
learned critics, multimodal inputs, colocated rollout, and multi-node jobs are
outside the supported configuration.

### Launch and inspect the run

```bash
export CONFIG_NAME=neuron_gsm8k
export RUN_DIR="$NEURON_EXAMPLE_DIR/runs/$(date +%Y%m%d-%H%M%S)"
bash "$PLUGIN_DIR/scripts/accelerators/neuron/run_neuron.sh"
```

The launcher checks the environments and devices, starts verl, and writes logs
to `$RUN_DIR/train.log`. It refuses an existing Ray cluster or occupied
NeuronCores. Cold native compilation can take more than 15 minutes before
training starts. Preserve the NKI and vLLM caches at their original paths:
compiled graphs contain absolute paths to NKI binaries.

Look for two completed training steps and their console metrics. This example
disables evaluation and disk checkpoints; actor weights still synchronize to
rollout through `neuron_nixl`. For a longer run with evaluation, pass Hydra
overrides to the same launcher:

```bash
RUN_DIR="$NEURON_EXAMPLE_DIR/runs/longer-$(date +%Y%m%d-%H%M%S)" \
  bash "$PLUGIN_DIR/scripts/accelerators/neuron/run_neuron.sh" \
  data.train_max_samples=-1 data.val_max_samples=32 \
  trainer.total_training_steps=20 trainer.test_freq=5
```

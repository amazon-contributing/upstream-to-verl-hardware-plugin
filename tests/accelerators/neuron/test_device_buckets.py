# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Exercise the registered sender and native-to-lite transfer through three policy versions."""

import asyncio
import json
import os
import socket
import subprocess
import sys
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def test_fused_qkv_shards_cross_head_and_bucket_boundaries():
    from verl_hardware_plugin.accelerators.neuron.weight_sync.device_buckets import DeviceBucketLayout, Shard

    source = "layers.0.attention.qkv_linear.wqkv.weight"
    names = {
        f"model.layers.{{}}.self_attn.{p}_proj.weight": f"layers.{{}}.attention.qkv_linear.w{p}.weight" for p in "qkv"
    }
    hf = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=2, tie_word_embeddings=False)
    raw = torch.arange(16 * 5).view(16, 5).float()
    # Include an empty shard and a replica on a different mesh. Each cut can fall inside a KV group.
    shards = [raw[:6], raw[6:12], raw[12:], raw[16:], raw[:6]]
    offsets = [0, 6, 12, 16, 0]
    metadata = [
        {source: Shard(tuple(raw.shape), (start, 0), tuple(shard.shape))}
        for start, shard in zip(offsets, shards, strict=True)
    ]
    layouts = [DeviceBucketLayout(metadata, rank, hf, names, bucket_bytes=32) for rank in range(len(shards))]
    actual = {}
    for index, bucket in enumerate(layouts[0].buckets):
        assembled = torch.zeros(bucket.buf_numel, dtype=torch.bfloat16)
        for layout, shard in zip(layouts, shards, strict=True):
            local = torch.zeros_like(assembled)
            for copy in layout.copies[index]:
                copy.copy(local, shard)
            assembled += local
        for entry in bucket.entries:
            actual.setdefault(entry.name, []).append(assembled.narrow(0, entry.offset, entry.numel))
    q, k, v = raw.view(2, 8, 5).split((4, 2, 2), dim=1)
    for p, ref in zip("qkv", (q, k, v), strict=True):
        assert torch.equal(torch.cat(actual[f"model.layers.0.self_attn.{p}_proj.weight"]), ref.flatten().bfloat16())


@pytest.mark.parametrize("end", [4, 6])
def test_incomplete_or_overlapping_actor_coverage(end):
    from verl_hardware_plugin.accelerators.neuron.weight_sync.device_buckets import Shard, owned_shards

    shards = [Shard((10,), (0,), (end,)), Shard((10,), (5,), (5,))]
    with pytest.raises(ValueError, match="actor shards"):
        owned_shards([{"norm": s} for s in shards], 0)


def test_hybrid_projections_from_actor_shards_to_native_rollout():
    from verl_hardware_plugin.accelerators.neuron.weight_sync.device_buckets import DeviceBucketLayout, Shard
    from verl_hardware_plugin.accelerators.neuron.weight_sync.streaming import validate_box_coverage
    from verl_hardware_plugin.accelerators.neuron.weight_sync.weight_layout import apply_box, pack_box, plan_box

    prefix = "layers.0.attn."
    weights = {}
    for part, rows in (("q", 4), ("k", 4), ("v", 12), ("z", 12)):
        values = torch.arange(rows * 6).view(rows, 6) + len(weights) * 100
        weights[prefix + f"in_proj_{part}.weight"] = values.bfloat16()
        if part != "z":
            weights[prefix + f"conv_{part}.weight"] = values[:, :4].reshape(rows, 1, 4).bfloat16()
    for name in ("A_log", "dt_bias"):
        weights[prefix + name] = torch.arange(12).float()
    adapter = {
        f"model.language_model.layers.{{}}.linear_attn.{name}": f"layers.{{}}.attn.{name}"
        for name in ("in_proj_z.weight", "A_log", "dt_bias")
    }
    metadata, locals_ = [], []
    for rank in range(2):
        shards, local = {}, {}
        for name, weight in weights.items():
            axis = weight.ndim - 1
            local[name] = weight.chunk(2, axis)[rank]
            offset = [0] * weight.ndim
            offset[axis] = rank * local[name].shape[axis]
            shards[name] = Shard(tuple(weight.shape), tuple(offset), tuple(local[name].shape))
        metadata.append(shards)
        locals_.append(local)
    hf = SimpleNamespace(text_config=SimpleNamespace(tie_word_embeddings=False))
    layouts = [DeviceBucketLayout(metadata, rank, hf, adapter, bucket_bytes=44) for rank in range(2)]
    shapes = {name: shape for name, shape, _ in layouts[0].schedule}
    buckets = layouts[0].buckets
    for rank in range(4):
        resident = "language_model.layers.0.linear_attn."
        qkvz = [weights[prefix + f"in_proj_{part}.weight"].chunk(4)[rank] for part in "qkvz"]
        conv = [weights[prefix + f"conv_{part}.weight"].chunk(4)[rank] for part in "qkv"]
        expected = {
            resident + "in_proj_qkvz_weight": torch.cat(qkvz).T.contiguous(),
            resident + "conv1d_weight": torch.cat(conv),
            **{resident + name: weights[prefix + name].chunk(4)[rank] for name in ("A_log", "dt_bias")},
        }
        inventory = {name: tuple(weight.shape) for name, weight in expected.items()}
        plans = [plan_box(bucket, shapes, inventory, rank, 4) for bucket in buckets]
        validate_box_coverage(plans, inventory)
        actual = {name: torch.full_like(weight, torch.nan) for name, weight in expected.items()}
        for index, (bucket, plan) in enumerate(zip(buckets, plans, strict=True)):
            buffer = torch.zeros(bucket.buf_numel, dtype=torch.bfloat16)
            for layout, local in zip(layouts, locals_, strict=True):
                contribution = torch.zeros_like(buffer)
                for copy in layout.copies[index]:
                    copy.copy(contribution, local[copy.source])
                buffer += contribution
            slot = torch.empty(plan.numel, dtype=torch.bfloat16)
            pack_box(slot, buffer, bucket, plan)
            apply_box(slot, plan, actual, {})
        for name, weight in expected.items():
            torch.testing.assert_close(actual[name], weight, rtol=0, atol=0)


def host_weights(step):
    generator = torch.Generator().manual_seed(31 + step)
    shapes = {"model.embed_tokens.weight": (20, 9), "model.norm.weight": (9,)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + norm + ".weight"] = (9,)
        for kind, rows in (("q", 16), ("k", 8), ("v", 8)):
            shapes[prefix + f"self_attn.{kind}_proj.weight"] = (rows, 9)
        shapes[prefix + "self_attn.o_proj.weight"] = (9, 16)
    weights = {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}
    weights["model.norm.weight"][0] = -0.0
    return weights


def expected_weights(step, rank):
    weights = {name: value.bfloat16() for name, value in host_weights(step).items()}
    expected = {"model.norm.weight": weights["model.norm.weight"]}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        expected[name] = weights["model.embed_tokens.weight"].chunk(4)[rank]
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            name = prefix + norm + ".weight"
            expected[name] = weights[name]
        attn = prefix + "self_attn."
        q = weights[attn + "q_proj.weight"].chunk(4)[rank]
        k, v = (weights[attn + f"{p}_proj.weight"].chunk(2)[rank // 2] for p in "kv")
        expected[attn + "qkv_proj_weight"] = torch.cat((q, k, v)).T.contiguous()
        expected[attn + "o_proj_weight"] = weights[attn + "o_proj.weight"].chunk(4, dim=1)[rank].T.contiguous()
    return expected


def register_weight(module, name, tensor):
    *path, leaf = name.split(".")
    for part in path:
        if not hasattr(module, part):
            module.add_module(part, torch.nn.Module())
        module = getattr(module, part)
    module.register_parameter(leaf, torch.nn.Parameter(tensor, requires_grad=False))


def actor_model(step, rank, mesh, adapter):
    from torch.distributed.tensor import DTensor, Replicate, Shard

    weights = host_weights(step)
    raw = adapter.from_hf_map
    native = {}
    for name, weight in weights.items():
        native[raw[name.replace(".0.", ".{}.").replace(".1.", ".{}.")].format(name.split(".")[2])] = weight
    prefix = "layers.0.attention.qkv_linear.w"
    q, k, v = (native.pop(prefix + p + ".weight").view(2, -1, 9) for p in "qkv")
    native[prefix + "qkv.weight"] = torch.cat((q, k, v), dim=1).reshape(-1, 9)
    model = torch.nn.Module()
    for name, weight in native.items():
        placement = (
            Replicate() if name.endswith("ffn_norm.weight") else Shard(1 if name == "tok_embeddings.weight" else 0)
        )
        local = weight if placement.is_replicate() else weight.chunk(2, dim=placement.dim)[rank]
        value = DTensor.from_local(
            local.contiguous().to("neuron:0"), mesh, [placement], shape=weight.shape, stride=weight.stride()
        )
        register_weight(model, name, value)
    model.lm_head = model.tok_embeddings
    return model


def guard_sender(stack):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    for owner, method in ((DTensor, "full_tensor"), (DTensor, "redistribute"), (torch.nn.Module, "state_dict")):
        stack.enter_context(patch.object(owner, method, side_effect=AssertionError(f"sender called {method}")))

    def metadata_only(gather):
        def guarded(output, tensor, *args, **kwargs):
            assert tensor.device.type == "cpu", "sender gathered device weights"
            return gather(output, tensor, *args, **kwargs)

        return guarded

    for method in ("all_gather", "all_gather_into_tensor"):
        stack.enter_context(patch.object(dist, method, metadata_only(getattr(dist, method))))


def run_actor(rank, store):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

    from verl_hardware_plugin.accelerators.neuron.engines import torchtitan_neuron as titan
    from verl_hardware_plugin.accelerators.neuron.weight_sync import checkpoint_engine
    from verl_hardware_plugin.accelerators.neuron.weight_sync.checkpoint_engine import NeuronNixlCheckpointEngine

    dist.init_process_group(backend="cpu:gloo,neuron:neuron")
    mesh = init_device_mesh("neuron", (2,))
    hf = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=4, tie_word_embeddings=True)
    adapter = Qwen3StateDictAdapter(None, None)
    actor = object.__new__(titan.TorchTitanNeuronEngineWithLMHead)
    actor.parallel_dims = SimpleNamespace(get_optional_mesh=lambda axis: mesh if axis == "tp" else None)
    actor.model_config = SimpleNamespace(hf_config=hf)
    actor.trainer = SimpleNamespace(device=torch.device("neuron:0"))
    actor.checkpointer = SimpleNamespace(sd_adapter=adapter)
    reference = torch.arange(65536).to(torch.int16)
    part = reference.clone()
    part[1 - rank :: 2] = 0
    buffer = part.view(torch.bfloat16).to("neuron:0")
    checkpoint_engine._reduce_bucket(buffer, [mesh.get_group()])
    assert torch.equal(buffer.view(torch.int16).cpu(), reference)
    for step in range(3):
        # Replace parameters entirely to catch cached values as well as stale views.
        actor.module = [actor_model(step, rank, mesh, adapter)]
        with ExitStack() as stack:
            guard_sender(stack)
            stack.enter_context(patch.object(titan, "_actor_engine", actor))
            stack.enter_context(patch.object(checkpoint_engine, "_ALL_REDUCE_TILE_BYTES", 64))
            if step == 0:
                sender = NeuronNixlCheckpointEngine(bucket_size=128, timeout_s=180)
            metadata = [None, None]
            dist.all_gather_object(metadata, sender.prepare(), group=sender.export_group)
            args, rollout = sender.build_topology(2, 1, [*metadata, {"rollout_tp": 4}])
            if rank == 0:
                payload = {key: values[0] for key, values in rollout.items()}
                payload["spec"] = vars(payload["spec"])
                store.set(f"prepare/{step}", json.dumps(payload))
            sender.init_process_group(args["spec"][rank])
            weights, _ = actor.get_per_tensor_param()
            asyncio.run(sender.send_weights(weights, global_steps=step))
    for endpoint in sender.endpoints.values():
        endpoint.close()
    dist.destroy_process_group()


def run_receiver(rank, store):
    from verl_hardware_plugin.accelerators.neuron.rollout.worker_extension import NeuronWorkerExtension
    from verl_hardware_plugin.accelerators.neuron.weight_sync.nixl_transport import NixlTopology

    worker = NeuronWorkerExtension()
    model = torch.nn.Module()
    for name, value in expected_weights(0, rank).items():
        register_weight(model, name, torch.full_like(value, torch.nan, device="neuron:0"))
    model.rank, model.world_size = rank, 4
    worker.rank, worker.device = rank, torch.device("neuron:0")
    worker.model_runner = SimpleNamespace(model=model)
    for step in range(3):
        rollout = json.loads(store.get(f"prepare/{step}"))
        if step == 0:
            worker.weight_sync_connect(NixlTopology(**rollout["spec"]).receiver(0))
        worker.weight_sync_recv(rollout["schedule"], rollout["bucket_bytes"])
        expected = expected_weights(step, rank)
        for name, value in model.named_parameters():
            torch.testing.assert_close(
                value.cpu().view(torch.int16), expected[name].view(torch.int16), atol=0, rtol=0, msg=name
            )
    worker.weight_sync_close()


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.skipif("NEURON_BUCKET_TEST_CORES" not in os.environ, reason="Requires two actor and four rollout cores")
def test_device_buckets_end_to_end(tmp_path):
    cores = os.environ["NEURON_BUCKET_TEST_CORES"].split(",")
    assert len(set(cores)) == len(cores) == 6
    master_port, store_port = (str(free_port()) for _ in range(2))
    processes = []
    try:
        for i, core in enumerate(cores):
            actor = i < 2
            role, rank = ("trainer", i) if actor else ("rollout", i - 2)
            env = {
                **os.environ,
                "VERL_PLATFORM": "aws",
                "VERL_NEURON_PROCESS_ROLE": role,
                "PYTHONPATH": os.pathsep.join(
                    [str(Path(__file__).resolve().parents[3]), os.environ.get("PYTHONPATH", "")]
                ),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": master_port,
                "RANK": str(rank),
                "WORLD_SIZE": "2" if actor else "4",
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "2" if actor else "4",
                "NEURON_RT_VISIBLE_CORES": core,
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "VLLM_PLUGINS": "",
                "NEURON_RT_MAP_HBM": "1",
                "NEURON_RT_ENABLE_P2P": "0",
                "FI_EFA_ENABLE_SHM_TRANSFER": "0",
                "NEURON_LOGICAL_NC_CONFIG": "2",
            }
            if not actor:
                # Standalone lite executors each own one core (vLLM sets this for its TP group).
                env["NEURON_RT_ROOT_COMM_ID"] = f"127.0.0.1:{free_port()}"
            log = tmp_path / f"{role}-{rank}.log"
            with log.open("w") as output:
                child = subprocess.Popen(
                    [sys.executable, __file__, role, str(rank), store_port],
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            processes.append((child, log))
        for child, log in processes:
            assert child.wait(timeout=360) == 0, log.read_text()
    finally:
        for child, _ in processes:
            if child.poll() is None:
                child.kill()
            child.wait()


if __name__ == "__main__":
    from importlib import import_module

    role, rank, port = sys.argv[1:]
    actor = role == "trainer"
    import_module("torch_neuronx" if actor else "libtorch_neuronx_lite")
    store = torch.distributed.TCPStore("127.0.0.1", int(port), 6, actor and rank == "0", timedelta(seconds=180))
    (run_actor if actor else run_receiver)(int(rank), store)

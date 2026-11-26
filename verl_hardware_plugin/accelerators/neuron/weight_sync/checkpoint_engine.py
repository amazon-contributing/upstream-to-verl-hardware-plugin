# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""NIXL-only checkpoint bridge for native TorchTitan and native vLLM-Neuron."""

import json
import logging
import os
import time
from datetime import timedelta
from functools import lru_cache

import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from ..runtime import physical_core
from .device_buckets import DeviceBucketLayout, local_parameters
from .nixl_transport import NixlEndpoint, NixlTopology, control_store
from .streaming import validate_box_coverage
from .weight_layout import pack_box, plan_box, slot_numel_for

# The pinned Trainium2 runtime rejects large all-reduces on multi-device groups
# (40 MiB channel limit). Tile the collective independently of the NIXL slot.
_ALL_REDUCE_TILE_BYTES = 32 * 1024 * 1024

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@lru_cache(maxsize=1)
def _compiled_word_cast():
    def cast(words, dtype):
        return words.to(dtype)

    return torch.compile(cast, backend="neuron", fullgraph=True)


def _reduce_bucket(buffer, groups):
    if not groups:
        return
    # INT16 collectives in the pinned runtime stage through host memory.
    # Signed BF16 words fit exactly in the native INT32 sum. Bitcast outside
    # the compiled conversions so Neuron preserves negative zero and NaN payloads.
    cast = _compiled_word_cast()
    for tile in buffer.split(_ALL_REDUCE_TILE_BYTES // 4):
        source = tile.clone() if tile.storage_offset() else tile
        words = cast(source.view(torch.int16), torch.int32)
        for group in groups:
            torch.distributed.all_reduce(words, group=group)
        tile.copy_(cast(words, torch.int16).view(torch.bfloat16))


@CheckpointEngineRegistry.register("neuron_nixl")
class NeuronNixlCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size, is_master=False, timeout_s=300, **kwargs):
        from ..engines.torchtitan_neuron import actor_engine

        self.engine = actor_engine()
        if self.engine.trainer.device.type != "neuron":
            raise ValueError("neuron_nixl requires real Trainium trainer tensors, not a CPU/Gloo fallback")
        self.bucket_size = int(bucket_size)
        self.timeout_s = float(timeout_s)
        self.connected = False
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = self.engine.trainer.device
        self.core = physical_core(os.environ["NEURON_RT_VISIBLE_CORES"], int(os.environ.get("LOCAL_RANK", "0")))
        self.endpoints, self.plans = {}, {}
        self.export_group = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=self.timeout_s))

        # A full actor pool can span a set of cores for which Neuron has no
        # single collective. Successive sums along the training mesh axes
        # reduce over the same Cartesian product using supported replica groups.
        axes = (self.engine.parallel_dims.get_optional_mesh(axis) for axis in ("tp", "fsdp", "dp_replicate"))
        self.device_groups = [mesh.get_group() for mesh in axes if mesh is not None and mesh.size() > 1]
        self.layout = DeviceBucketLayout.from_engine(self.engine, self.bucket_size, self.export_group)
        self.schedule = self.layout.schedule

    def prepare(self):
        import ray

        metadata = {"role": "actor", "schedule": self.schedule, "bucket_size": self.bucket_size}
        if self.rank == 0:
            host = ray.util.get_node_ip_address()
            if getattr(self, "master_store", None) is None:
                # Bind port 0 and keep the listening store. Probing a free port, closing it and binding it
                # again in init_process_group races with other processes on the host (EADDRINUSE).
                self.master_store = control_store(
                    NixlTopology(host, 0, rollout_tp=1, replicas=1, timeout_s=self.timeout_s), master=True
                )
                self.port = self.master_store.port
            metadata.update(host=host, port=self.port, timeout_s=self.timeout_s)
        return metadata

    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        actors, replicas = metadata[:actor_wg_world_size], metadata[actor_wg_world_size:]
        master = actors[0]
        if len(replicas) != rollout_world_size or not replicas:
            raise ValueError("NIXL requires at least one rollout replica")
        if any(item["schedule"] != master["schedule"] for item in actors):
            raise ValueError("Trainer ranks disagree on weight metadata")
        if any(item["bucket_size"] != master["bucket_size"] for item in actors):
            raise ValueError("Trainer ranks disagree on the NIXL bucket size")
        degrees = {item["rollout_tp"] for item in replicas}
        if len(degrees) != 1:
            raise ValueError("All rollout replicas must use the same TP degree")
        spec = NixlTopology(
            store_host=master["host"],
            store_port=master["port"],
            rollout_tp=degrees.pop(),
            timeout_s=master["timeout_s"],
            replicas=rollout_world_size,
        )
        actor_args = {"spec": [spec] * actor_wg_world_size}
        rollout_args = {
            "spec": [spec] * rollout_world_size,
            "schedule": [master["schedule"]] * rollout_world_size,
            "bucket_bytes": [master["bucket_size"]] * rollout_world_size,
            "replica_index": list(range(rollout_world_size)),
        }
        return actor_args, rollout_args

    def init_process_group(self, spec):
        if self.connected:
            if self.spec != spec:
                raise ValueError("NIXL topology cannot change during this run")
            return
        self.spec = spec
        store = getattr(self, "master_store", None) if self.rank == 0 else None
        self.store = store if store is not None else control_store(spec, master=self.rank == 0)
        shapes = {name: shape for name, shape, dtype in self.schedule}
        for replica in range(spec.replicas):
            for rank in range(spec.rollout_tp):
                if (replica * spec.rollout_tp + rank) % self.world_size != self.rank:
                    continue
                prefix = f"replica-{replica}-tp-{rank}"
                endpoint = NixlEndpoint(
                    self.store, prefix, self.device, self.core, sender=True, timeout_s=spec.timeout_s
                )
                self.endpoints[replica, rank] = endpoint
                inventory = {name: tuple(shape) for name, shape in json.loads(endpoint.store.get("inventory")).items()}
                plans = [plan_box(bucket, shapes, inventory, rank, spec.rollout_tp) for bucket in self.layout.buckets]
                validate_box_coverage(plans, inventory)
                self.plans[replica, rank] = plans
        self.connected = True

    async def send_weights(self, weights, global_steps=None):
        if not self.connected:
            raise RuntimeError("Connect both NIXL pools before sending weights")
        started = time.monotonic()
        buckets = self.layout.buckets
        local = local_parameters(weights, self.device)
        for key, endpoint in self.endpoints.items():
            endpoint.register(slot_numel_for(self.plans[key]))
        buffer = torch.empty(buckets[0].buf_numel, dtype=torch.bfloat16, device=self.device)
        for index, bucket in enumerate(buckets):
            buffer.zero_()
            for copy in self.layout.copies[index]:
                copy.copy(buffer, local[copy.source])
            _reduce_bucket(buffer, self.device_groups)
            for key, endpoint in self.endpoints.items():
                plan = self.plans[key][index]
                if plan.numel:
                    pack_box(endpoint.buffer, buffer, bucket, plan)
                    endpoint.push(plan.numel)  # waits for this receiver to apply the slot
        torch.distributed.barrier(group=self.export_group)
        logger.info(
            "neuron_nixl rank %d exported %d HF buckets at policy step %s for %d TP receivers",
            self.rank,
            len(buckets),
            global_steps,
            len(self.endpoints),
        )
        return {"timing_s/neuron_nixl/send": time.monotonic() - started} if self.rank == 0 else {}

    async def receive_weights(self, global_steps=None):
        raise RuntimeError("NIXL receivers run inside native vLLM workers")

    def finalize(self):
        return None

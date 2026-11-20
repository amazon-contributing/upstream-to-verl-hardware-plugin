# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Assemble one HF bucket in shared host memory for a single-node actor pool."""

import os
import tempfile
from math import prod

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard


def local_boxes(exporter):
    boxes = {}
    for parts in exporter.sources.values():
        for name in parts:
            value = exporter.state[name]
            shape = tuple(value.shape)
            if isinstance(value, DTensor):
                if not all(isinstance(p, Shard | Replicate | _StridedShard) for p in value.placements):
                    raise ValueError(f"Cannot export partial sums for {name}")
                if value.to_local().numel() == 0:
                    continue
                size, offset = compute_local_shape_and_global_offset(shape, value.device_mesh, value.placements)
                if size != tuple(value.to_local().shape):
                    raise ValueError(f"DTensor shape does not match local shard of {name}")
            else:
                size, offset = shape, (0,) * len(shape)
            boxes[name] = (tuple(offset), tuple(size))
    return boxes


def select_writers(exporter, rank_boxes):
    """Deduplicate replicas and prove that disjoint native shards cover each tensor."""
    owners = {}
    for parts in exporter.sources.values():
        for name in parts:
            shape = tuple(exporter.state[name].shape)
            volumes = {}
            for rank, boxes in enumerate(rank_boxes):
                if name not in boxes:
                    continue
                offset, size = boxes[name]
                if (offset, size) in volumes:
                    continue
                if len(size) != len(shape) or any(
                    n <= 0 or i < 0 or i + n > s for i, n, s in zip(offset, size, shape, strict=True)
                ):
                    raise ValueError(f"Invalid shard for {name}")
                for start, extent in volumes:
                    if all(a < b + m and b < a + n for a, n, b, m in zip(offset, size, start, extent, strict=True)):
                        raise ValueError(f"Overlapping shards for {name}")
                volumes[offset, size] = rank
            if sum(prod(size) for _, size in volumes) != prod(shape):
                raise ValueError(f"Incomplete shard coverage for {name}")
            owners[name] = volumes
    return owners


class SharedExport:
    def __init__(self, exporter, capacity, group):
        rank = dist.get_rank(group)
        boxes = local_boxes(exporter)
        all_boxes = [None] * dist.get_world_size(group)
        dist.all_gather_object(all_boxes, boxes, group=group)
        owners = select_writers(exporter, all_boxes)
        self.owned = {name: box for name, box in boxes.items() if owners[name][box] == rank}
        # The file is private to this job, unlinked as soon as every rank maps it.
        name = [None, None]
        if rank == 0:
            try:
                fd, name[0] = tempfile.mkstemp(prefix="verl-qwen-", dir="/dev/shm")
                try:
                    os.posix_fallocate(fd, 0, capacity * 2)
                finally:
                    os.close(fd)
            except OSError as error:
                name[1] = str(error)
                if name[0] is not None:
                    os.unlink(name[0])
        dist.broadcast_object_list(name, src=0, group=group)
        if name[1]:
            raise RuntimeError(f"Failed to allocate shared weight bucket: {name[1]}")
        try:
            shared = [None] * len(all_boxes)
            dist.all_gather_object(shared, os.path.isfile(name[0]), group=group)
            if not all(shared):
                raise ValueError("The actor pool must share /dev/shm for weight_sync_shared_host_bucket")
            self.frame = torch.from_file(name[0], shared=True, size=capacity, dtype=torch.bfloat16)
            dist.barrier(group=group)
        finally:
            if rank == 0:
                os.unlink(name[0])

    def buckets(self, exporter, buckets):
        last_key, local = None, None
        for bucket in buckets:
            for entry in bucket.entries:
                width = prod(entry.shape[1:])
                first = entry.param_offset // width
                stop = first + entry.numel // width
                fragment = self.frame.narrow(0, entry.offset, entry.numel).view(-1, *entry.shape[1:])
                base = 0
                for key in exporter.sources[entry.name]:
                    value = exporter.state[key]
                    if key in self.owned:
                        offset, size = self.owned[key]
                        start = base + offset[0]
                        lo, hi = max(first, start), min(stop, start + size[0])
                        if lo < hi:
                            if key != last_key:
                                shard = value.to_local() if isinstance(value, DTensor) else value
                                local = shard.detach().to("cpu").to(torch.bfloat16)
                                last_key = key
                            tail = tuple(slice(i, i + n) for i, n in zip(offset[1:], size[1:], strict=True))
                            fragment[(slice(lo - first, hi - first), *tail)].copy_(local[lo - start : hi - start])
                    base += value.shape[0]
            # Caller barriers: publish all writes before packing, then wait for
            # every sender to receive its ACK before reusing this single frame.
            yield self.frame

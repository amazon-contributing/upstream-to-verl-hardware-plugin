# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Map live dense TorchTitan shards to bounded BF16 buffers in HF row order."""

import re
from dataclasses import dataclass
from math import prod

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from .weight_layout import plan_buckets


@dataclass(frozen=True)
class Shard:
    shape: tuple
    offsets: tuple
    sizes: tuple


def describe(parameter):
    shape = tuple(parameter.shape)
    if isinstance(parameter, DTensor):
        # DCP accounts for uneven chunks and FSDP's strided placement over TP.
        (chunk,) = parameter.__create_chunk_list__()
        return Shard(shape, tuple(chunk.offsets), tuple(chunk.sizes))
    return Shard(shape, (0,) * len(shape), shape)


def owned_shards(metadata, rank):
    """Elect one contributor for each box, including replicas on distinct submeshes."""
    result = {}
    if any(item.keys() != metadata[0].keys() for item in metadata):
        raise ValueError("Actor ranks disagree on live parameter names")
    for name, local in metadata[rank].items():
        owners = {}
        for peer, descriptions in enumerate(metadata):
            shard = descriptions[name]
            if shard.shape != local.shape or any(
                start < 0 or size < 0 or start + size > extent
                for start, size, extent in zip(shard.offsets, shard.sizes, shard.shape, strict=True)
            ):
                raise ValueError(f"Invalid actor shard for {name}")
            if prod(shard.sizes):
                owners.setdefault(shard, peer)
        boxes = []
        for shard in owners:
            box = tuple((start, start + size) for start, size in zip(shard.offsets, shard.sizes, strict=True))
            if any(
                all(lo < end and start < hi for (lo, hi), (start, end) in zip(box, other, strict=True))
                for other in boxes
            ):
                raise ValueError(f"Overlapping actor shards for {name}")
            boxes.append(box)
        if sum(prod(shard.sizes) for shard in owners) != prod(local.shape):
            raise ValueError(f"Incomplete actor shards for {name}")
        if owners.get(local) == rank:
            result[name] = local
    return result


def projection_rows(name, shape, config):
    """Yield native name, HF shape, and (native row, HF row, count) runs."""
    if ".qkv_linear.wqkv." not in name:
        yield name, shape, [(0, 0, shape[0])]
        return
    heads = config.num_key_value_heads
    head_dim = config.head_dim
    q_rows = config.num_attention_heads // heads * head_dim
    group_rows = q_rows + 2 * head_dim
    if shape[0] != heads * group_rows:
        raise ValueError(f"Fused QKV shape disagrees with attention config for {name}")
    for kind, offset, count in (("q", 0, q_rows), ("k", q_rows, head_dim), ("v", q_rows + head_dim, head_dim)):
        runs = [(head * group_rows + offset, head * count, count) for head in range(heads)]
        yield name.replace(".wqkv.", f".w{kind}."), (heads * count, *shape[1:]), runs


def hf_projections(shards, hf, from_hf_map):
    """Describe each HF tensor by row runs in its native actor parameters."""
    names = {native: hf_name for hf_name, native in from_hf_map.items() if native}
    gdn = {}
    for source, shard in shards.items():
        name = source.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
        if hf.tie_word_embeddings and name == "lm_head.weight":
            continue  # plan_box maps the embedding to both resident destinations.
        match = re.fullmatch(r"layers\.(\d+)\.attn\.(in_proj|conv)_([qkv])\.weight", name)
        if match:
            layer, member, part = match.groups()
            fused = "conv1d" if member == "conv" else "in_proj_qkv"
            parts = gdn.setdefault(f"model.layers.{layer}.linear_attn.{fused}.weight", {})
            if part in parts:
                raise ValueError(f"Duplicate GDN component {name}")
            parts[part] = source
            continue
        for native, shape, row_runs in projection_rows(name, shard.shape, hf):
            template = re.sub(r"(?<=layers\.)\d+", "{}", native)
            layer = re.search(r"layers\.(\d+)", native)
            hf_name = names[template].format(layer[1] if layer else "")
            yield hf_name.replace("model.language_model.", "model.", 1), shape, {source: row_runs}

    for name, parts in gdn.items():
        if parts.keys() != set("qkv"):
            raise ValueError(f"Incomplete GDN projection {name}")
        tail = shards[parts["q"]].shape[1:]
        offset, runs = 0, {}
        for part in "qkv":
            source = parts[part]
            shape = shards[source].shape
            if shape[1:] != tail:
                raise ValueError(f"Incompatible GDN component shapes for {name}")
            runs[source] = [(0, offset, shape[0])]
            offset += shape[0]
        yield name, (offset, *tail), runs


@dataclass(frozen=True)
class CopyRows:
    source: str
    local_row: int
    rows: int
    offset: int
    width: int
    column: int
    columns: int

    def copy(self, buffer, local):
        source = local[self.local_row : self.local_row + self.rows]
        if source.storage_offset() and source.dtype != torch.bfloat16:
            # Bound the cast to the current bucket and compile it at offset zero.
            source = source.clone()
        source = source.to(torch.bfloat16)
        if self.columns != self.width:
            source = F.pad(source.reshape(self.rows, self.columns), (self.column, self.width - self.column - self.columns))
        buffer.narrow(0, self.offset, self.rows * self.width).copy_(source.reshape(-1))


def _bucket_copies(bucket, runs, owned):
    """Intersect each bucket entry with this rank's HF row runs."""
    copies = []
    for entry in bucket.entries:
        width = prod(entry.shape[1:])
        first = entry.param_offset // width
        for source, hf_row, local_row, count in runs[entry.name]:
            begin, end = max(first, hf_row), min(first + entry.numel // width, hf_row + count)
            if begin < end:
                shard = owned[source]
                copies.append(
                    CopyRows(
                        source,
                        local_row + begin - hf_row,
                        end - begin,
                        entry.offset + (begin - first) * width,
                        width,
                        shard.offsets[-1] if len(shard.shape) > 1 else 0,
                        prod(shard.sizes[1:]),
                    )
                )
    return copies


class DeviceBucketLayout:
    """Cached actor copy instructions. All offsets refer to the live local weights."""

    @classmethod
    def from_engine(cls, engine, bucket_bytes, group):
        parameters, _ = engine.get_per_tensor_param()
        local = {name: describe(value) for name, value in parameters}
        metadata = [None] * dist.get_world_size(group)
        dist.all_gather_object(metadata, local, group=group)
        return cls(
            metadata,
            dist.get_rank(group),
            engine.model_config.hf_config,
            engine.checkpointer.sd_adapter.from_hf_map,
            bucket_bytes,
        )

    def __init__(self, metadata, rank, hf, from_hf_map, bucket_bytes):
        owned = owned_shards(metadata, rank)
        shards = metadata[rank]
        text = getattr(hf, "text_config", hf)
        shapes, runs = {}, {}
        for hf_name, shape, sources in hf_projections(shards, text, from_hf_map):
            if hf_name in shapes:
                raise ValueError(f"Duplicate actor weight {hf_name}")
            shapes[hf_name] = shape
            runs[hf_name] = []
            for source, row_runs in sources.items():
                shard = shards[source]
                if source not in owned:
                    continue
                for native_row, hf_row, count in row_runs:
                    begin = max(shard.offsets[0], native_row)
                    end = min(shard.offsets[0] + shard.sizes[0], native_row + count)
                    if begin < end:
                        runs[hf_name].append(
                            (source, hf_row + begin - native_row, begin - shard.offsets[0], end - begin)
                        )
        self.schedule = [(name, shapes[name], "bfloat16") for name in sorted(shapes)]
        self.buckets = plan_buckets(self.schedule, bucket_bytes=bucket_bytes)
        self.copies = [_bucket_copies(bucket, runs, owned) for bucket in self.buckets]


def local_parameters(named_parameters, device):
    values = {}
    for name, value in named_parameters:
        local = value.to_local() if isinstance(value, DTensor) else value
        if local.device.type != device.type or (device.index is not None and local.device.index != device.index):
            raise ValueError(f"Expected live actor weight {name} on {device}, got {local.device}")
        values[name] = local.detach()
    return values

# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Stream HF weights without constructing full GDN projections in Trainium HBM."""

import re

import torch
from torch.distributed.tensor import DTensor


class QwenTextExport:
    """Map native parameters to the HF wire; gather only the requested parameter."""

    def __init__(self, state, adapter):
        self.state = state
        self.sources = {}
        direct = {native: hf for hf, native in adapter.from_hf_map.items() if native is not None}
        groups = {}
        for name in state:
            abstract = re.sub(r"(?<=layers\.)\d+", "{}", name, count=1)
            layer = re.match(r"layers\.(\d+)\.", name)
            if abstract in direct:
                hf = direct[abstract].format(layer[1]) if layer else direct[abstract]
                self.sources[hf.replace("model.language_model.", "model.", 1)] = (name,)
                continue
            fused = re.fullmatch(r"layers\.(\d+)\.attn\.(in_proj|conv)_([qkv])\.weight", name)
            if fused:
                member = "in_proj_qkv" if fused[2] == "in_proj" else "conv1d"
                target = f"model.layers.{fused[1]}.linear_attn.{member}.weight"
                groups.setdefault(target, {})[fused[3]] = name
        for target, pieces in groups.items():
            self.sources[target] = tuple(pieces[part] for part in "qkv")

    def metadata(self):
        result = []
        for name in sorted(self.sources):
            parts = self.sources[name]
            first = self.state[parts[0]].shape
            shape = (sum(self.state[key].shape[0] for key in parts), *first[1:])
            result.append((name, shape, "bfloat16"))
        return result

    def tensors(self, names=None):
        for name in sorted(self.sources) if names is None else names:
            parts = []
            for key in self.sources[name]:
                # Move the local master shard off device *before* casting or
                # gathering. Adam state remains resident after the first step.
                part = self.state[key].detach().to("cpu").to(torch.bfloat16)
                if isinstance(part, DTensor):
                    part = part.full_tensor()
                parts.append(part)
            yield name, (torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]).contiguous()

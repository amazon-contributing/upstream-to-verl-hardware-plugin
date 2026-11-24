# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Verify coverage of the native rollout parameters."""


def validate_box_coverage(plans, inventory):
    """Reject skipped exports and incomplete resident weight coverage."""
    from math import prod

    counts = {}
    boxes = {}
    for plan in plans:
        if plan.skipped:
            raise ValueError(f"Rollout cannot apply parameters: {plan.skipped}")
        for piece in plan.pieces:
            shape = inventory[piece.name]
            if len(piece.dst_box) != len(shape) or any(
                lower < 0 or upper > extent or lower >= upper
                for (lower, upper), extent in zip(piece.dst_box, shape, strict=True)
            ):
                raise ValueError(f"Invalid rollout weight box for {piece.name}")
            previous_boxes = boxes.setdefault(piece.name, [])
            for previous in previous_boxes:
                if all(
                    max(lower, other_lower) < min(upper, other_upper)
                    for (lower, upper), (other_lower, other_upper) in zip(piece.dst_box, previous, strict=True)
                ):
                    raise ValueError(f"Overlapping rollout weight coverage for {piece.name}")
            previous_boxes.append(piece.dst_box)
            counts[piece.name] = counts.get(piece.name, 0) + piece.numel
    for name, shape in inventory.items():
        if counts.get(name, 0) != prod(shape):
            raise ValueError(f"Incomplete or overlapping rollout weight coverage for {name}")

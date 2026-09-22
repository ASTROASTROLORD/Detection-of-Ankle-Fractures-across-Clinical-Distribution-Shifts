"""Geometry-preserving image preprocessing metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResizePlan:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale: float
    pad_left: int
    pad_top: int

    def map_box_to_source(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        return tuple(
            round(value, 6)
            for value in (
                (x1 - self.pad_left) / self.scale,
                (y1 - self.pad_top) / self.scale,
                (x2 - self.pad_left) / self.scale,
                (y2 - self.pad_top) / self.scale,
            )
        )


def make_resize_plan(width: int, height: int, longest_edge: int = 1024) -> ResizePlan:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(1.0, longest_edge / max(width, height))
    target_width = round(width * scale)
    target_height = round(height * scale)
    return ResizePlan(
        width,
        height,
        target_width,
        target_height,
        scale,
        (longest_edge - target_width) // 2,
        (longest_edge - target_height) // 2,
    )

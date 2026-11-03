"""Geometry metadata for the manuscript's MMDetection resize and padding."""

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
                (x1 - self.pad_left) * self.source_width / self.target_width,
                (y1 - self.pad_top) * self.source_height / self.target_height,
                (x2 - self.pad_left) * self.source_width / self.target_width,
                (y2 - self.pad_top) * self.source_height / self.target_height,
            )
        )

    @property
    def padded_width(self) -> int:
        return (self.target_width + 31) // 32 * 32

    @property
    def padded_height(self) -> int:
        return (self.target_height + 31) // 32 * 32


def make_resize_plan(width: int, height: int, longest_edge: int = 1333,
                     shortest_edge: int = 800) -> ResizePlan:
    """Mirror Resize(keep_ratio=True) followed by right/bottom divisor-32 padding.

    The scale is the nominal resize factor. Coordinate inversion uses the
    actual rounded width and height ratios, matching MMDetection metadata.
    """
    if min(width, height, longest_edge, shortest_edge) <= 0:
        raise ValueError("image dimensions and resize bounds must be positive")
    if shortest_edge > longest_edge:
        raise ValueError("shortest edge must not exceed longest edge")
    scale = min(shortest_edge / min(width, height), longest_edge / max(width, height))
    target_width = max(1, int(width * scale + 0.5))
    target_height = max(1, int(height * scale + 0.5))
    return ResizePlan(
        width,
        height,
        target_width,
        target_height,
        scale,
        0,
        0,
    )

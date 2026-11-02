"""Typed representation of a single YOLO object detection."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

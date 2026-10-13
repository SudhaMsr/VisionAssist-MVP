"""Thin wrapper around the existing YOLOv8n checkpoint.

Returns plain dataclasses so the guidance rules can be tested without a model.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    # Pixel coordinates in the original image: x1, y1, x2, y2.
    bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int

    @property
    def center_x_frac(self) -> float:
        return ((self.bbox[0] + self.bbox[2]) / 2) / self.image_width

    @property
    def area_frac(self) -> float:
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        return (w * h) / (self.image_width * self.image_height)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [round(v, 1) for v in self.bbox]
        d["confidence"] = round(self.confidence, 4)
        return d


@dataclass
class FrameResult:
    detections: list[Detection]
    annotated: np.ndarray  # BGR image with boxes drawn
    inference_ms: float


class Detector:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.35,
        imgsz: int = 640,
        device: str | None = None,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not 0.0 < conf < 1.0:
            raise ValueError(f"conf must be in (0, 1), got {conf}")

        from ultralytics import YOLO  # imported lazily: heavy dependency

        start = time.perf_counter()
        self.model = YOLO(str(model_path))
        self.load_ms = (time.perf_counter() - start) * 1000
        self.model_path = model_path
        self.names: dict[int, str] = dict(self.model.names)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: str | Path | np.ndarray) -> FrameResult:
        if isinstance(image, (str, Path)):
            import cv2

            path = Path(image)
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Could not read image: {path}")
        else:
            frame = image
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            raise ValueError("Expected a non-empty HxWx3 BGR image")

        start = time.perf_counter()
        result = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        inference_ms = (time.perf_counter() - start) * 1000

        h, w = frame.shape[:2]
        detections = []
        if result.boxes is not None:
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                detections.append(
                    Detection(
                        class_name=self.names[int(cls)],
                        confidence=float(conf),
                        bbox=tuple(float(v) for v in xyxy),
                        image_width=w,
                        image_height=h,
                    )
                )
        return FrameResult(detections=detections, annotated=result.plot(), inference_ms=inference_ms)

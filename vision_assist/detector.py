"""Thin wrapper around the existing YOLOv8n checkpoint.

Returns plain dataclasses so the guidance rules can be tested without a model.

Optionally, a general-purpose COCO model can assist with the classes both models
know (vehicles and people), which the ROD model misses on footage unlike its
training data. ROD detections are always kept; an assist detection is only added
when it does not duplicate one already found. All other ROD classes (stairs,
manholes, poles, ...) come from the ROD model alone.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# COCO class -> ROD class, for the classes both models detect.
COCO_TO_ROD = {
    "car": "Car", "person": "Person", "bus": "Bus", "truck": "Truck", "motorcycle": "Motorcycle", "bicycle": "Bike",
}
SHARED_CLASSES = frozenset(COCO_TO_ROD.values())
ASSIST_COLOUR = (255, 128, 0)  # BGR: assist boxes are drawn in a different colour from the ROD model's


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    # Pixel coordinates in the original image: x1, y1, x2, y2.
    bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    source: str = "rod"  # "rod" (main model) or "assist" (COCO model)

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
        if self.source == "rod":  # keep output unchanged when no assist model is used
            del d["source"]
        return d


def _overlap(a: tuple, b: tuple) -> tuple[float, float]:
    """(intersection over union, intersection over the smaller box)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def merge_assist(
    primary: list[Detection], assist: list[Detection], iou: float = 0.5, ios: float = 0.8
) -> list[Detection]:
    """Keep every primary detection; add assist detections that are not duplicates.

    An assist box is a duplicate if it overlaps (IoU >= `iou`, or covers >= `ios` of the
    smaller box) any primary detection of a shared class, or an assist box already added.
    The class is ignored when matching, so the same vehicle is not reported as both
    "car" and "truck".
    """
    kept = [d for d in primary if d.class_name in SHARED_CLASSES]
    added: list[Detection] = []
    for det in sorted(assist, key=lambda d: -d.confidence):
        if det.class_name not in SHARED_CLASSES:
            continue
        if any(o[0] >= iou or o[1] >= ios for o in (_overlap(det.bbox, k.bbox) for k in kept + added)):
            continue
        added.append(det)
    return [*primary, *added]


def _load_yolo(path: Path):
    try:
        from ultralytics import YOLO  # imported lazily: heavy dependency
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed: pip install -r requirements.txt") from exc
    try:
        return YOLO(str(path))
    except Exception as exc:  # ultralytics raises TypeError/RuntimeError/pickle errors for bad files
        reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        raise RuntimeError(f"Could not load checkpoint {path}: {reason}") from exc


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
        assist_model_path: str | Path | None = None,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if assist_model_path is not None and not Path(assist_model_path).is_file():
            raise FileNotFoundError(f"Assist model not found: {assist_model_path}")
        if not 0.0 < conf < 1.0:
            raise ValueError(f"conf must be in (0, 1), got {conf}")

        start = time.perf_counter()
        self.model = _load_yolo(model_path)
        self.assist = _load_yolo(Path(assist_model_path)) if assist_model_path is not None else None
        self.load_ms = (time.perf_counter() - start) * 1000
        self.model_path = model_path
        self.assist_model_path = Path(assist_model_path) if assist_model_path is not None else None
        if self.assist is not None:
            missing = set(COCO_TO_ROD) - set(self.assist.names.values())
            if missing:
                raise ValueError(f"Assist model lacks the shared COCO classes {sorted(missing)}")
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
        inference_ms = (time.perf_counter() - start) * 1000  # model time only, as before
        detections = self._detections(result, self.names, frame, "rod")
        annotated = result.plot()
        if self.assist is not None:
            start = time.perf_counter()
            assist_result = self.assist.predict(
                source=frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
            )[0]
            inference_ms += (time.perf_counter() - start) * 1000
            coco = [
                Detection(COCO_TO_ROD[d.class_name], d.confidence, d.bbox, d.image_width, d.image_height, "assist")
                for d in self._detections(assist_result, assist_result.names, frame, "assist")
                if d.class_name in COCO_TO_ROD
            ]
            detections = merge_assist(detections, coco)
            for det in detections:
                if det.source == "assist":
                    _draw_box(annotated, det)
        return FrameResult(detections=detections, annotated=annotated, inference_ms=inference_ms)

    @staticmethod
    def _detections(result, names: dict, frame: np.ndarray, source: str) -> list[Detection]:
        h, w = frame.shape[:2]
        detections = []
        if result.boxes is not None:
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                detections.append(
                    Detection(
                        class_name=names[int(cls)],
                        confidence=float(conf),
                        bbox=tuple(float(v) for v in xyxy),
                        image_width=w,
                        image_height=h,
                        source=source,
                    )
                )
        return detections


def _draw_box(image: np.ndarray, det: Detection) -> None:
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
    thickness = max(2, round(image.shape[1] / 400))
    cv2.rectangle(image, (x1, y1), (x2, y2), ASSIST_COLOUR, thickness)
    label = f"{det.class_name} {det.confidence:.2f} (assist)"
    scale = max(0.5, image.shape[1] / 1400)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), ASSIST_COLOUR, -1)
    cv2.putText(image, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)

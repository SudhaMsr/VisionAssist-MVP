"""ROD + COCO assist: merging rules, and the real two-model detector."""

from pathlib import Path

import pytest

from vision_assist.config import DEFAULT_MODEL_PATH
from vision_assist.detector import SHARED_CLASSES, Detection, Detector, merge_assist

ASSIST_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"


def det(name, box, conf=0.8, source="rod"):
    return Detection(name, conf, box, 600, 400, source)


def test_all_primary_detections_kept_in_order():
    primary = [det("Stairs", (0, 0, 100, 100)), det("Car", (200, 200, 300, 300)), det("Manhole", (10, 300, 90, 390))]
    assert merge_assist(primary, [])[:3] == primary
    merged = merge_assist(primary, [det("Car", (400, 0, 500, 100), source="assist")])
    assert merged[:3] == primary and merged[3].source == "assist"


def test_same_object_is_not_reported_twice():
    rod_car = det("Car", (100, 100, 200, 200))
    assist = [
        det("Car", (105, 102, 205, 198), source="assist"),  # same car (IoU about 0.9)
        det("Car", (120, 120, 180, 180), source="assist"),  # box inside the ROD box
        det("Car", (400, 100, 500, 200), source="assist"),  # a different car
    ]
    merged = merge_assist([rod_car], assist)
    assert [d.bbox for d in merged] == [rod_car.bbox, (400, 100, 500, 200)]


def test_cross_class_duplicate_dropped():
    # The ROD model calls it a truck, COCO calls it a car: still one vehicle.
    merged = merge_assist([det("Truck", (100, 100, 300, 300))], [det("Car", (110, 110, 300, 290), source="assist")])
    assert [d.class_name for d in merged] == ["Truck"]


def test_assist_duplicates_among_themselves_keep_highest_confidence():
    a = det("Car", (100, 100, 200, 200), conf=0.6, source="assist")
    b = det("Car", (102, 101, 201, 199), conf=0.9, source="assist")
    assert merge_assist([], [a, b]) == [b]


def test_overlap_with_rod_only_class_does_not_suppress():
    # A person standing on stairs is a separate hazard, not a duplicate.
    merged = merge_assist([det("Stairs", (0, 0, 600, 400))], [det("Person", (250, 100, 300, 300), source="assist")])
    assert [d.class_name for d in merged] == ["Stairs", "Person"]


def test_non_shared_assist_classes_ignored():
    assert merge_assist([], [det("Bench", (0, 0, 50, 50), source="assist")]) == []
    assert SHARED_CLASSES == {"Car", "Person", "Bus", "Truck", "Motorcycle", "Bike"}


def test_to_dict_marks_only_assist_detections():
    assert "source" not in det("Car", (0, 0, 1, 1)).to_dict()
    assert det("Car", (0, 0, 1, 1), source="assist").to_dict()["source"] == "assist"


def test_missing_assist_model_rejected(tmp_path):
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    with pytest.raises(FileNotFoundError, match="Assist model"):
        Detector(DEFAULT_MODEL_PATH, assist_model_path=tmp_path / "nope.pt")


def test_assist_model_without_coco_classes_rejected():
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    with pytest.raises(ValueError, match="lacks the shared COCO classes"):
        Detector(DEFAULT_MODEL_PATH, assist_model_path=DEFAULT_MODEL_PATH)


@pytest.fixture(scope="module")
def ensemble():
    pytest.importorskip("ultralytics")
    if not (DEFAULT_MODEL_PATH.is_file() and ASSIST_MODEL.is_file()):
        pytest.skip("ROD checkpoint or models/yolov8n.pt missing")
    return Detector(DEFAULT_MODEL_PATH, device="cpu", assist_model_path=ASSIST_MODEL)


def test_real_ensemble_keeps_rod_output_and_adds_only_shared_classes(ensemble, detector, ultralytics_assets):
    image = ultralytics_assets / "bus.jpg"
    rod_only = detector.detect(image).detections
    merged = ensemble.detect(image)
    assert merged.detections[: len(rod_only)] == rod_only
    added = merged.detections[len(rod_only):]
    assert added, "COCO should add something on bus.jpg (the ROD model misses the bus)"
    assert all(d.source == "assist" and d.class_name in SHARED_CLASSES for d in added)
    assert "Bus" in {d.class_name for d in added}
    assert merged.annotated.shape[:2] == (rod_only[0].image_height, rod_only[0].image_width)

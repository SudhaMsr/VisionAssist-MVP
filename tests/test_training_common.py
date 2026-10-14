"""training.common: shared geometry/IO helpers used across the fine-tuning pipeline."""

import pytest

from training.common import (
    COCO_TO_ROD, COMPLETE_IN_ROD, PROTECTED_CLASSES, ROD_ONLY_CLASSES, CLASS_NAMES,
    overlap, read_list, read_yolo_labels, to_xyxy, yolo_line,
)


def test_class_mapping_is_internally_consistent():
    assert set(COCO_TO_ROD.values()) <= set(CLASS_NAMES)
    assert set(COMPLETE_IN_ROD) <= set(COCO_TO_ROD.values())
    assert "bicycle" not in COCO_TO_ROD  # ROD "Bike" deliberately left unmapped (mixed bicycles/motorbikes)
    assert set(ROD_ONLY_CLASSES) == set(CLASS_NAMES) - set(COCO_TO_ROD.values())
    assert set(PROTECTED_CLASSES) <= set(CLASS_NAMES)


def test_read_list_strips_comments_and_blanks(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("# header\na.jpg  # note\n\nb.jpg\n")
    assert read_list(f) == ["a.jpg", "b.jpg"]


def test_yolo_line_round_trips_through_to_xyxy():
    box = (10.0, 20.0, 110.0, 70.0)
    line = yolo_line(2, box, w=200, h=100)
    cls, cx, cy, bw, bh = line.split()
    recovered = to_xyxy((int(cls), float(cx), float(cy), float(bw), float(bh)), 200, 100)
    assert recovered == pytest.approx(box, abs=0.1)


def test_yolo_line_clips_to_image_and_rejects_degenerate_boxes():
    assert yolo_line(0, (-50, -50, 2000, 2000), w=100, h=100) is not None  # clipped, not dropped
    assert yolo_line(0, (10, 10, 10.5, 50), w=100, h=100) is None  # under 1px wide after rounding


def test_overlap_iou_and_containment():
    a, b = (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)
    assert overlap(a, b) == (1.0, 1.0)
    iou, ios = overlap((0.0, 0.0, 10.0, 10.0), (2.0, 2.0, 4.0, 4.0))  # b fully inside a
    assert iou == pytest.approx(4 / 100) and ios == 1.0
    assert overlap((0, 0, 1, 1), (5, 5, 6, 6)) == (0.0, 0.0)


def test_read_yolo_labels_converts_polygons_like_training(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("2 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n\n")
    rows = read_yolo_labels(f)
    assert len(rows) == 2 and rows[0][0] == 2 and rows[1][0] == 1


def test_read_yolo_labels_missing_file_returns_empty(tmp_path):
    assert read_yolo_labels(tmp_path / "missing.txt") == []

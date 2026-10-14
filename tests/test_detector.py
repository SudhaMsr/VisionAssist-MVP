"""Smoke tests against the real checkpoint. Skipped if ultralytics or the model is missing.

These check output *shape and validity*, not accuracy: accuracy is reported from
the recorded validation run in results/rod_yolov8n_50ep/metrics.json.
"""

import statistics
from pathlib import Path

import numpy as np
import pytest
import yaml

from vision_assist.detector import Detector

REPO = Path(__file__).resolve().parent.parent

# Generous CPU budget: this is a regression tripwire, not a real-time claim.
CPU_MEDIAN_INFERENCE_BUDGET_MS = 1000


def test_checkpoint_classes_match_config(detector):
    names = yaml.safe_load((REPO / "configs" / "rod_detection.yaml").read_text())["names"]
    assert [detector.names[i] for i in range(len(detector.names))] == names


def test_detections_are_well_formed(detector, ultralytics_assets):
    result = detector.detect(ultralytics_assets / "bus.jpg")
    h, w = result.annotated.shape[:2]
    assert result.detections, "expected at least one detection on bus.jpg"
    for d in result.detections:
        assert d.class_name in detector.names.values()
        assert detector.conf <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert (d.image_width, d.image_height) == (w, h)


def test_people_found_on_sample_image(detector, ultralytics_assets):
    # bus.jpg shows several people and a bus. The ROD checkpoint finds the people
    # but NOT the large side-on bus behind them, even at conf 0.1 (out-of-domain). That
    # miss is recorded as a known limitation in the README; it is not asserted
    # here because the test should not lock in a model weakness.
    found = [d.class_name for d in detector.detect(ultralytics_assets / "bus.jpg").detections]
    assert found.count("Person") >= 2, found


def test_blank_image_gives_no_detections(detector):
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)).detections == []


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Detector(tmp_path / "nope.pt")


def test_invalid_confidence_rejected(detector):
    with pytest.raises(ValueError):
        Detector(detector.model_path, conf=1.5)


def test_unreadable_image_raises(detector, tmp_path):
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Could not read"):
        detector.detect(bad)


def test_wrong_array_shape_raises(detector):
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))


def test_cpu_inference_latency(detector, ultralytics_assets, record_property):
    image = ultralytics_assets / "bus.jpg"
    detector.detect(image)  # warm-up
    times = [detector.detect(image).inference_ms for _ in range(10)]
    median = statistics.median(times)
    record_property("median_inference_ms", round(median, 1))
    print(f"\nCPU inference over 10 runs: median {median:.1f} ms, max {max(times):.1f} ms")
    assert median < CPU_MEDIAN_INFERENCE_BUDGET_MS


def test_corrupt_checkpoint_raises_clear_error(tmp_path):
    pytest.importorskip("ultralytics")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Could not load checkpoint"):
        Detector(bad)


def test_checkpoint_error_without_message_is_still_reported(tmp_path, monkeypatch):
    ultralytics = pytest.importorskip("ultralytics")

    def broken_loader(*args, **kwargs):
        raise Exception()

    monkeypatch.setattr(ultralytics, "YOLO", broken_loader)
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Could not load checkpoint .*: Exception"):
        Detector(ckpt)

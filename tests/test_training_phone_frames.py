"""training.phone_frames: walk-video frame extraction for labelling."""

import cv2
import numpy as np
import pytest

from training.phone_frames import extract


def make_video(path, seconds=3, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(seconds * fps):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def test_extract_writes_one_frame_per_second_and_empty_label_folder(tmp_path):
    video = make_video(tmp_path / "walk01_test.mp4")
    result = extract(video, tmp_path / "out", fps=1.0)
    assert result["status"] == "written" and result["frames"] == 3
    images = sorted((tmp_path / "out" / "walk01_test" / "images").iterdir())
    assert [p.name for p in images] == ["walk01_test_0001.jpg", "walk01_test_0002.jpg", "walk01_test_0003.jpg"]
    assert (tmp_path / "out" / "walk01_test" / "labels").is_dir()


def test_extract_rejects_bad_walk_names(tmp_path):
    video = make_video(tmp_path / "walk 01 (final).mp4")
    with pytest.raises(ValueError, match="letters, digits"):
        extract(video, tmp_path / "out", fps=1.0)


def test_extract_skips_an_existing_walk_without_overwriting(tmp_path):
    video = make_video(tmp_path / "walk02.mp4")
    out = tmp_path / "out"
    extract(video, out, fps=1.0)
    (out / "walk02" / "images" / "sentinel.txt").write_text("keep me")
    result = extract(video, out, fps=1.0)
    assert result["status"] == "exists, skipped"
    assert (out / "walk02" / "images" / "sentinel.txt").read_text() == "keep me"

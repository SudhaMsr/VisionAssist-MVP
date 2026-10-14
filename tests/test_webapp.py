"""Web UI: helper functions, plus headless runs of app.py with Streamlit's AppTest."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_assist.config import DEFAULT_MODEL_PATH
from vision_assist.webapp import (
    alert_rows,
    detection_rows,
    expected_frames,
    frame_caption,
    list_videos,
    new_run_dir,
    save_upload,
    zip_outputs,
)

REPO = Path(__file__).resolve().parent.parent


def write_video(path, image, seconds=3, fps=10):
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(seconds * fps):
        writer.write(image)
    writer.release()
    return path


def test_list_videos_only_mp4_sorted(tmp_path):
    for name in ("b.mp4", "a.MP4", "notes.txt", "c.avi"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_videos([tmp_path, tmp_path / "missing"])] == ["a.MP4", "b.mp4"]


def test_new_run_dir_is_unique_and_safe(tmp_path):
    a = new_run_dir("../my clip!.mp4", tmp_path)
    b = new_run_dir("../my clip!.mp4", tmp_path)
    assert a != b and a.parent == tmp_path and b.parent == tmp_path
    assert a.name.endswith("my_clip") and a.is_dir() and b.is_dir()


def test_save_upload_validates(tmp_path):
    assert save_upload("walk.mp4", b"data", tmp_path).read_bytes() == b"data"
    with pytest.raises(ValueError, match=".mp4"):
        save_upload("walk.mov", b"data", tmp_path)
    with pytest.raises(ValueError, match="empty"):
        save_upload("walk.mp4", b"", tmp_path)


def test_expected_frames_matches_sampling(tmp_path):
    from vision_assist.stream import frames_from_video

    video = write_video(tmp_path / "v.mp4", np.zeros((48, 64, 3), np.uint8), seconds=3, fps=10)
    for fps, limit in ((2, None), (1, None), (2, 4)):
        assert expected_frames(video, fps, limit) == len(list(frames_from_video(video, fps, limit)))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Could not open"):
        expected_frames(bad, 2, None)


SUMMARY = {
    "per_frame": [
        {"index": 1, "t_s": 0.0, "status": "spoken", "message": "Person, centre.", "interrupted": None,
         "spoken": [{"tier": "medium"}], "detections": [
             {"class_name": "Person", "confidence": 0.8, "bbox": [270, 100, 330, 400], "image_width": 600,
              "image_height": 400},
             {"class_name": "Road", "confidence": 0.9, "bbox": [0, 300, 600, 400], "image_width": 600,
              "image_height": 400}]},
        {"index": 2, "t_s": 0.5, "status": "spoken", "message": "Caution. Car, left.",
         "interrupted": "Person, centre.", "spoken": [{"tier": "high"}], "detections": []},
        {"index": 3, "t_s": 1.0, "status": "busy", "message": None, "would_say": "Dog, right.",
         "spoken": [], "detections": []},
    ]
}


def test_alert_rows_mark_priority_and_cut_offs():
    rows = alert_rows(SUMMARY)
    assert [(r["message"], r["priority"], r["cut off later"]) for r in rows] == [
        ("Person, centre.", "normal", True), ("Caution. Car, left.", "urgent", False)]
    assert rows[1]["cut off what was playing"] == "Person, centre."


def test_detection_rows_add_tier_and_region():
    rows = detection_rows(SUMMARY)
    assert [(r["class"], r["tier"], r["image region"]) for r in rows] == [
        ("Person", "medium", "centre"), ("Road", "ignored", "centre")]


def test_frame_caption_wording():
    assert frame_caption(SUMMARY["per_frame"][1]) == "Spoke: Caution. Car, left. (cut off: Person, centre.)"
    assert frame_caption(SUMMARY["per_frame"][2]) == "Still speaking; held back: Dog, right."


def test_zip_outputs_excludes_upload(tmp_path):
    (tmp_path / "input.mp4").write_bytes(b"x")
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "frame_0001.jpg").write_bytes(b"j")
    (tmp_path / "transcript.txt").write_text("t")
    names = zipfile.ZipFile(io.BytesIO(zip_outputs(tmp_path))).namelist()
    assert sorted(names) == ["frames/frame_0001.jpg", "transcript.txt"]


@pytest.fixture
def ui_env(tmp_path, monkeypatch, fake_engine_exe):
    videos, runs = tmp_path / "videos", tmp_path / "runs"
    videos.mkdir()
    monkeypatch.setenv("VISION_ASSIST_VIDEOS", str(videos))
    monkeypatch.setenv("VISION_ASSIST_UI_OUTPUT", str(runs))
    monkeypatch.setenv("VISION_ASSIST_TTS", fake_engine_exe)
    return videos, runs


def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(REPO / "app.py"), default_timeout=180)


def test_app_analyses_a_chosen_video(ui_env, ultralytics_assets):
    pytest.importorskip("streamlit")
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    videos, runs = ui_env
    write_video(videos / "bus.mp4", cv2.imread(str(ultralytics_assets / "bus.jpg")), seconds=3)

    at = app_test().run()
    assert not at.exception
    at.radio[0].set_value("Choose an existing MP4").run()
    assert at.selectbox[0].value.name == "bus.mp4"
    at.button[0].click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]

    (run_dir,) = runs.iterdir()
    summary = json.loads((run_dir / "stream_results.json").read_text())
    assert summary["frames"] == 6  # 3 s at 2 frames per second
    assert summary["metrics"]["overlapping_messages"] == 0
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Frames analysed"] == "6" and int(metrics["Messages spoken"]) >= 1
    assert any("Person" in str(df.value) for df in at.dataframe)
    downloads = [b for b in at.get("download_button")]
    assert len(downloads) == 5
    for name in ("vision_assist_stream.mp4", "vision_assist_stream.wav", "transcript.txt", "stream_results.json"):
        assert (run_dir / name).is_file()


def test_app_reports_unreadable_video(ui_env):
    pytest.importorskip("streamlit")
    videos, runs = ui_env
    (videos / "broken.mp4").write_bytes(b"not a video")
    at = app_test().run()
    at.radio[0].set_value("Choose an existing MP4").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Could not open video" in e.value for e in at.error)
    assert not at.metric


def test_app_without_videos_explains_how_to_get_one(ui_env):
    pytest.importorskip("streamlit")
    at = app_test().run()
    at.radio[0].set_value("Choose an existing MP4").run()
    assert any("make_sample_video.py" in i.value for i in at.info)
    assert at.button[0].disabled

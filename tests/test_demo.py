"""End-to-end demo run on the ultralytics sample images."""

import json

import pytest

from conftest import real_tts_available
from vision_assist.config import DEFAULT_MODEL_PATH, list_images, resolve_image_dir
from vision_assist.demo import find_ffmpeg, main, probe_media, run_demo
from vision_assist.speech import find_tts_engine, wav_duration


@pytest.fixture
def engine(fake_engine):
    return find_tts_engine() if real_tts_available() else fake_engine


def test_end_to_end_outputs(tmp_path, ultralytics_assets, engine):
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    images = [ultralytics_assets / "bus.jpg", ultralytics_assets / "zidane.jpg", ultralytics_assets / "bus.jpg"]
    summary = run_demo(images, DEFAULT_MODEL_PATH, tmp_path, tts_engine=engine, device="cpu", cooldown_s=60)

    for name in ("vision_assist_demo.wav", "vision_assist_demo.mp4", "demo_results.json", "transcript.txt"):
        assert (tmp_path / name).is_file(), name
    assert len(list((tmp_path / "frames").glob("frame_*.jpg"))) == 3

    frames = summary["frames"]
    assert frames[0]["message"], "bus.jpg contains people; expected an alert"
    assert len(frames[0]["alerts_spoken"]) <= 2
    # Same image again within the cooldown: nothing spoken for frame 1 may repeat.
    key = lambda a: (a["class_name"], a["image_region"])
    first = {key(a) for a in frames[0]["alerts_spoken"]}
    assert not first & {key(a) for a in frames[2]["alerts_spoken"]}
    assert first <= {key(a) for a in frames[2]["alerts_suppressed"]}

    total = sum(f["duration_s"] for f in frames)
    assert wav_duration(tmp_path / "vision_assist_demo.wav") == pytest.approx(total, abs=0.01)
    probe = summary["outputs"]["video_probe"]
    assert probe["has_video"] and probe["has_audio"]
    assert probe["duration_s"] == pytest.approx(total, abs=0.25)

    saved = json.loads((tmp_path / "demo_results.json").read_text())
    assert saved["latency"]["inference"]["n"] == 3


def test_cli_reports_missing_images(tmp_path, capsys):
    assert main(["--images", str(tmp_path / "missing"), "--out", str(tmp_path)]) == 1
    assert "No input images" in capsys.readouterr().err


def test_list_images_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(resolve_image_dir(tmp_path))


def test_probe_detects_audio_stream(tmp_path, fake_engine):
    from vision_assist.speech import synthesize

    synthesize("Car, left.", tmp_path / "a.wav", fake_engine)
    probe = probe_media(tmp_path / "a.wav", find_ffmpeg())
    assert probe["has_audio"] and not probe["has_video"]


def test_read_image_list_keeps_order_and_ignores_comments(tmp_path):
    from vision_assist.config import read_image_list

    for name in ("b.jpg", "a.jpg"):
        (tmp_path / name).write_bytes(b"x")
    listing = tmp_path / "list.txt"
    listing.write_text("# header\nb.jpg  # Car, Person\n\na.jpg\n")
    assert [p.name for p in read_image_list(listing, tmp_path)] == ["b.jpg", "a.jpg"]


def test_read_image_list_reports_missing_images(tmp_path):
    from vision_assist.config import read_image_list

    listing = tmp_path / "list.txt"
    listing.write_text("missing.jpg\n")
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        read_image_list(listing, tmp_path)

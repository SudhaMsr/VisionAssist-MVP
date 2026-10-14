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


def test_list_images_rejects_video_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError, match="no video"):
        list_images(video)


def _write_jpgs(directory, names):
    import cv2
    import numpy as np

    for name in names:
        cv2.imwrite(str(directory / name), np.zeros((64, 64, 3), dtype=np.uint8))


def test_select_images_defaults_to_curated_list_when_present(tmp_path):
    from vision_assist.config import DEFAULT_IMAGE_LIST, read_image_list, select_images

    names = [line.split("#", 1)[0].strip() for line in DEFAULT_IMAGE_LIST.read_text().splitlines()]
    names = [n for n in names if n]
    _write_jpgs(tmp_path, names + ["AAA_first_sorted.jpg"])
    images, how = select_images(tmp_path)
    assert [p.name for p in images] == names and "curated" in how
    # An explicit limit means "first N in sorted order", as before.
    images, _ = select_images(tmp_path, limit=1)
    assert [p.name for p in images] == ["AAA_first_sorted.jpg"]
    assert select_images(tmp_path, image_list=DEFAULT_IMAGE_LIST)[0] == read_image_list(DEFAULT_IMAGE_LIST, tmp_path)


def test_select_images_falls_back_to_first_eight(tmp_path):
    from vision_assist.config import select_images

    _write_jpgs(tmp_path, [f"img_{i:02d}.jpg" for i in range(10)])
    images, how = select_images(tmp_path)
    assert [p.name for p in images] == [f"img_{i:02d}.jpg" for i in range(8)] and "first 8" in how


def test_unreadable_image_fails_before_any_output(tmp_path, fake_engine):
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    _write_jpgs(tmp_path, ["good.jpg"])
    bad.write_bytes(b"not an image")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="could not be read"):
        run_demo([good, bad], DEFAULT_MODEL_PATH, out, tts_engine=fake_engine, make_video=False)
    assert not out.exists()


def test_invalid_settings_fail_before_any_output(tmp_path, fake_engine):
    _write_jpgs(tmp_path, ["a.jpg"])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="max_alerts"):
        run_demo([tmp_path / "a.jpg"], DEFAULT_MODEL_PATH, out, tts_engine=fake_engine, max_alerts=0)
    assert not out.exists()


def test_rerun_removes_stale_outputs(tmp_path, ultralytics_assets, fake_engine):
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    bus, zidane = ultralytics_assets / "bus.jpg", ultralytics_assets / "zidane.jpg"
    run_demo([bus, zidane], DEFAULT_MODEL_PATH, tmp_path, tts_engine=fake_engine, device="cpu")
    assert (tmp_path / "vision_assist_demo.mp4").is_file()
    summary = run_demo([zidane], DEFAULT_MODEL_PATH, tmp_path, tts_engine=fake_engine, device="cpu", make_video=False)
    assert [p.name for p in (tmp_path / "frames").iterdir()] == ["frame_01.jpg"]
    spoken = [f for f in summary["frames"] if f["message"]]
    assert len(list((tmp_path / "speech_clips").iterdir())) == len(spoken)
    assert not (tmp_path / "vision_assist_demo.mp4").exists(), "stale MP4 would not match the new run"



@pytest.mark.parametrize("limit", [0, -1])
def test_select_images_rejects_non_positive_limit(tmp_path, limit):
    from vision_assist.config import select_images

    _write_jpgs(tmp_path, ["a.jpg", "b.jpg"])
    with pytest.raises(ValueError, match="limit"):
        select_images(tmp_path, limit=limit)


def test_cli_reports_unwritable_output_dir(tmp_path, capsys, ultralytics_assets, fake_engine_exe):
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")
    code = main(["--images", str(ultralytics_assets / "zidane.jpg"), "--out", str(not_a_dir),
                 "--tts", fake_engine_exe, "--no-video", "--device", "cpu"])
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("error:") and "Traceback" not in err


def test_failed_run_leaves_no_stale_results(tmp_path, capsys, ultralytics_assets, fake_engine_exe, failing_engine_exe):
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    args = ["--images", str(ultralytics_assets / "zidane.jpg"), "--out", str(tmp_path / "o"), "--no-video",
            "--device", "cpu"]
    assert main([*args, "--tts", fake_engine_exe]) == 0
    assert (tmp_path / "o" / "demo_results.json").is_file()
    assert main([*args, "--tts", failing_engine_exe]) == 1
    assert "TTS failed" in capsys.readouterr().err
    leftovers = [p.name for p in (tmp_path / "o").rglob("*") if p.is_file()]
    assert not {"demo_results.json", "transcript.txt", "vision_assist_demo.wav"} & set(leftovers), leftovers

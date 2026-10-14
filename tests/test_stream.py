"""Stream timing: one speech channel, real timestamps, unchanged alert rules."""

import json

import cv2
import numpy as np
import pytest

from vision_assist.config import DEFAULT_MODEL_PATH
from vision_assist.guidance import HIGH, LOW, MEDIUM, Alert, AlertScheduler, compose_message
from vision_assist.speech import wav_duration
from vision_assist.stream import StreamTimer, _truncate_wav, frames_from_video, main, run_stream


def alert(name, region="centre", tier=MEDIUM, area=0.1):
    return Alert(class_name=name, tier=tier, region=region, confidence=0.9, area_frac=area)


PERSON = alert("Person")
CAR = alert("Car", "left", HIGH)
DOG = alert("Dog", "right")


def run(frames, speech_s, fps=1.0, gap_s=0.5, cooldown_s=6.0, urgent_interrupts=True):
    """Feed per-frame alert lists through a StreamTimer; every message lasts `speech_s`."""
    timer = StreamTimer(AlertScheduler(cooldown_s=cooldown_s), gap_s=gap_s, urgent_interrupts=urgent_interrupts)
    outcomes = [timer.step(i / fps, alerts, lambda message: speech_s) for i, alerts in enumerate(frames)]
    return timer, outcomes


def test_channel_never_busy_gives_exactly_the_slideshow_rules():
    frames = [[PERSON, DOG], [PERSON], [CAR, PERSON], [], [DOG, CAR, PERSON], [PERSON], [PERSON], [CAR]]
    timer, outcomes = run(frames, speech_s=0.0, gap_s=0.0, cooldown_s=3.0)
    reference = AlertScheduler(cooldown_s=3.0)
    for i, (alerts, outcome) in enumerate(zip(frames, outcomes)):
        expected, _ = reference.select(alerts, now_s=float(i))
        assert outcome["message"] == compose_message(expected)
        assert outcome["status"] != "busy"


def test_speech_never_overlaps_and_busy_frames_are_not_spoken():
    timer, outcomes = run([[PERSON, DOG]] * 10, speech_s=2.4)
    assert [o["status"] for o in outcomes[:4]] == ["spoken", "busy", "busy", "quiet"]  # free again at 2.9 s
    starts = [e["start_s"] for e in timer.events]
    assert starts == [0.0, 6.0]  # cooldown, unchanged: re-announced after 6 s
    assert timer.metrics(10.0)["overlapping_messages"] == 0


def test_without_interrupts_new_hazard_waits_and_delay_is_measured():
    frames = [[PERSON], [PERSON, CAR], [PERSON, CAR], [PERSON, CAR]]
    timer, outcomes = run(frames, speech_s=2.4, urgent_interrupts=False)
    assert outcomes[1]["status"] == "busy" and outcomes[1]["would_say"] == "Caution. Car, left."
    assert outcomes[3]["message"] == "Caution. Car, left."
    car = outcomes[3]["spoken"][0]
    assert car["queue_delay_s"] == 2.0 and car["since_first_seen_s"] == 2.0
    urgent = timer.metrics(4.0)["urgent"]
    assert urgent["first_announcement_latency"] == {"n": 1, "median_s": 2.0, "max_s": 2.0}
    assert urgent["announced"] == 1


def test_without_interrupts_hazard_gone_before_channel_frees_is_dropped_not_spoken_late():
    frames = [[PERSON], [PERSON, CAR], [PERSON], [PERSON]]
    timer, outcomes = run(frames, speech_s=2.4, urgent_interrupts=False)
    assert all(o["message"] is None for o in outcomes[1:])
    metrics = timer.metrics(4.0)
    assert metrics["dropped_while_busy"] == 1
    assert metrics["urgent"]["not_announced_blocked_by_speech"] == 1
    assert timer.dropped[0]["class_name"] == "Car"


def test_urgent_alert_cuts_off_lower_tier_speech():
    frames = [[PERSON], [PERSON, CAR], [PERSON, CAR]]
    timer, outcomes = run(frames, speech_s=2.4)
    assert outcomes[1]["status"] == "spoken" and outcomes[1]["interrupted"] == "Person, centre."
    assert outcomes[1]["message"] == "Caution. Car, left."  # Person is still in its cooldown: rules unchanged
    assert outcomes[2]["status"] == "busy"
    first, second = timer.events
    assert first["end_s"] == 1.0 and first["interrupted"] and not second["interrupted"]
    m = timer.metrics(3.0)
    assert m["overlapping_messages"] == 0 and m["urgent_interruptions"] == 1 and m["speech_cut_s"] == 1.4
    assert m["urgent"]["first_announcement_latency"] == {"n": 1, "median_s": 0.0, "max_s": 0.0}


def test_urgent_alert_does_not_cut_off_urgent_speech():
    motorcycle = alert("Motorcycle", "right", HIGH)
    _, outcomes = run([[CAR], [CAR, motorcycle]], speech_s=2.4)
    assert outcomes[1]["status"] == "busy" and outcomes[1]["would_say"] == "Caution. Motorcycle, right."


def test_non_urgent_alert_never_interrupts():
    _, outcomes = run([[PERSON], [PERSON, DOG]], speech_s=2.4)
    assert outcomes[1]["status"] == "busy"


def test_urgent_alert_in_trailing_gap_starts_without_cutting_anything():
    timer, outcomes = run([[PERSON], [PERSON, CAR]], speech_s=0.8, gap_s=0.5)  # busy until 1.3 s
    assert outcomes[1]["message"] == "Caution. Car, left." and outcomes[1]["interrupted"] is None
    assert timer.events[0]["end_s"] == 0.8 and not timer.events[0]["interrupted"]


def test_urgent_alert_in_its_cooldown_does_not_interrupt():
    # Car announced at 0 s and goes out of view; Person starts at 3 s; Car returns at 4 s,
    # still inside its 6 s cooldown, so the unchanged rules would not say it: no cut-off.
    frames = [[CAR], [], [], [PERSON], [PERSON, CAR]]
    _, outcomes = run(frames, speech_s=2.4)
    assert outcomes[3]["message"] == "Person, centre."
    assert outcomes[4]["status"] == "busy" and outcomes[4]["would_say"] is None


def test_truncate_wav_cuts_and_fades(tmp_path):
    import wave

    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(1000)
        w.writeframes(np.full(2000, 10000, dtype="<i2").tobytes())
    _truncate_wav(path, 0.5)
    assert wav_duration(path) == 0.5
    with wave.open(str(path), "rb") as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert samples[0] == 10000 and samples[-1] == 0 and samples[-10] < 10000


def test_waiting_hazard_still_present_at_end_is_reported():
    timer, _ = run([[PERSON], [PERSON, DOG]], speech_s=5.0)
    assert timer.metrics(2.0)["still_waiting_at_end"] == 1


def test_low_tier_spoken_when_free():
    sign = alert("Traffic sign", "right", LOW)
    timer, outcomes = run([[sign]], speech_s=1.0)
    assert outcomes[0]["message"] == "Traffic sign, right."


def test_negative_gap_rejected():
    with pytest.raises(ValueError):
        StreamTimer(AlertScheduler(), gap_s=-1)


def test_video_is_sampled_at_requested_fps(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(30):  # 3 s at 10 fps; brightness encodes the source frame number
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    frames = list(frames_from_video(path, fps=2))
    assert [t for t, _, _ in frames] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert [round(float(f.mean()) / 8) for _, _, f in frames] == [0, 5, 10, 15, 20, 25]
    assert len(list(frames_from_video(path, fps=2, limit=2))) == 2


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Could not open video"):
        list(frames_from_video(bad, fps=1))


def test_run_stream_outputs_are_consistent(tmp_path, ultralytics_assets, fake_engine):
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("checkpoint missing")
    image = cv2.imread(str(ultralytics_assets / "bus.jpg"))
    fps = 3  # a frame period that is not a whole number of 10 fps video frames
    frames = [(i / fps, f"bus_{i}", image) for i in range(7)]
    summary = run_stream(frames, DEFAULT_MODEL_PATH, tmp_path, fps=fps, tts_engine=fake_engine, device="cpu")

    m = summary["metrics"]
    assert m["overlapping_messages"] == 0 and m["messages_spoken"] >= 1
    statuses = [f["status"] for f in summary["per_frame"]]
    assert statuses[0] == "spoken" and "busy" in statuses
    last_end = max(f["t_s"] for f in summary["per_frame"] if f["status"] == "spoken")
    audio_s = wav_duration(tmp_path / "vision_assist_stream.wav")
    assert audio_s >= 7 / fps - 0.01 and audio_s > last_end
    probe = summary["outputs"]["video_probe"]
    assert probe["has_audio"] and probe["has_video"]
    assert probe["duration_s"] == pytest.approx(audio_s, abs=0.15)
    assert len(list((tmp_path / "frames").glob("frame_*.jpg"))) == 7
    assert len((tmp_path / "transcript.txt").read_text().splitlines()) == m["messages_spoken"]
    assert json.loads((tmp_path / "stream_results.json").read_text())["metrics"] == m


def test_cli_rejects_bad_fps_and_missing_frames(tmp_path, capsys):
    assert main(["--image-list", str(tmp_path / "none.txt"), "--images", str(tmp_path), "--fps", "0"]) == 1
    assert "fps must be > 0" in capsys.readouterr().err
    listing = tmp_path / "list.txt"
    listing.write_text("missing.jpg\n")
    assert main(["--image-list", str(listing), "--images", str(tmp_path), "--out", str(tmp_path / "o")]) == 1
    assert "missing.jpg" in capsys.readouterr().err

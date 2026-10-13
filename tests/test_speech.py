import wave

import pytest

from conftest import real_tts_available
from vision_assist.speech import (
    SpeechError,
    TTSUnavailableError,
    assemble_timeline,
    find_tts_engine,
    synthesize,
    wav_duration,
)


def test_synthesize_writes_wav(tmp_path, fake_engine):
    out = tmp_path / "a.wav"
    synthesize("Caution. Stairs, centre.", out, fake_engine)
    assert wav_duration(out) > 0


def test_longer_text_gives_longer_audio(tmp_path, fake_engine):
    synthesize("Car, left.", tmp_path / "s.wav", fake_engine)
    synthesize("Caution. Stairs, centre. Car, left.", tmp_path / "l.wav", fake_engine)
    assert wav_duration(tmp_path / "l.wav") > wav_duration(tmp_path / "s.wav")


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_rejected(tmp_path, fake_engine, text):
    with pytest.raises(ValueError):
        synthesize(text, tmp_path / "x.wav", fake_engine)


def test_engine_failure_raises_with_stderr(tmp_path, fake_engine):
    with pytest.raises(SpeechError, match="simulated engine failure"):
        synthesize("FAIL", tmp_path / "x.wav", fake_engine)


def test_engine_writing_nothing_is_an_error(tmp_path, fake_engine):
    with pytest.raises(SpeechError, match="no audio"):
        synthesize("EMPTY", tmp_path / "x.wav", fake_engine)


def test_missing_engine_reports_install_hint():
    with pytest.raises(TTSUnavailableError, match="not found"):
        find_tts_engine("definitely-not-a-real-tts-binary")


def test_timeline_places_clips_and_pads_silence(tmp_path, fake_engine):
    clip = tmp_path / "c.wav"
    synthesize("Car, left.", clip, fake_engine)  # 0.5 s
    total = assemble_timeline([clip, None, clip], [1.0, 2.0, 1.5], tmp_path / "t.wav")
    assert total == pytest.approx(4.5)
    assert wav_duration(tmp_path / "t.wav") == pytest.approx(4.5)


def test_timeline_all_silent(tmp_path):
    assert assemble_timeline([None, None], [1.0, 1.0], tmp_path / "t.wav") == pytest.approx(2.0)


def test_timeline_rejects_clip_longer_than_slot(tmp_path, fake_engine):
    clip = tmp_path / "c.wav"
    synthesize("Caution. Stairs, centre. Car, left.", clip, fake_engine)
    with pytest.raises(ValueError, match="does not fit"):
        assemble_timeline([clip], [0.5], tmp_path / "t.wav")


def test_timeline_rejects_mixed_formats(tmp_path, fake_engine):
    a = tmp_path / "a.wav"
    synthesize("Car.", a, fake_engine)
    b = tmp_path / "b.wav"
    with wave.open(str(b), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    with pytest.raises(SpeechError, match="Incompatible"):
        assemble_timeline([a, b], [1.0, 1.0], tmp_path / "t.wav")


@pytest.mark.skipif(not real_tts_available(), reason="espeak-ng not installed")
def test_real_espeak_produces_speech(tmp_path):
    out = tmp_path / "real.wav"
    synthesize("Caution. Stairs, centre.", out, find_tts_engine())
    assert 0.5 < wav_duration(out) < 5.0

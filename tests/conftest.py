import shutil
import sys
from pathlib import Path

import pytest

from vision_assist.config import DEFAULT_MODEL_PATH

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def fake_engine() -> list[str]:
    return [sys.executable, str(TESTS_DIR / "fake_tts.py")]


@pytest.fixture(scope="session")
def ultralytics_assets() -> Path:
    """Sample photos bundled with the ultralytics package (bus.jpg, zidane.jpg).

    Real images, but NOT from the ROD dataset: they are used only to check the
    pipeline runs and outputs are well-formed, never to judge accuracy.
    """
    ultralytics = pytest.importorskip("ultralytics")
    assets = Path(ultralytics.__file__).parent / "assets"
    if not (assets / "bus.jpg").is_file():
        pytest.skip("ultralytics sample images not available")
    return assets


@pytest.fixture(scope="session")
def detector():
    pytest.importorskip("ultralytics")
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip(f"checkpoint missing: {DEFAULT_MODEL_PATH}")
    from vision_assist.detector import Detector

    return Detector(DEFAULT_MODEL_PATH, device="cpu")


def real_tts_available() -> bool:
    return bool(shutil.which("espeak-ng") or shutil.which("espeak"))


def _write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def fake_engine_exe(tmp_path) -> str:
    """The fake engine as a standalone executable, for code that looks engines up by path."""
    return str(_write_script(tmp_path / "fake-espeak", f'exec "{sys.executable}" "{TESTS_DIR / "fake_tts.py"}" "$@"'))


@pytest.fixture
def failing_engine_exe(tmp_path) -> str:
    """An engine that always fails, like a broken espeak-ng install."""
    return str(_write_script(tmp_path / "broken-espeak", 'echo "simulated engine failure" >&2; exit 3'))

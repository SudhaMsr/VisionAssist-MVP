# Vision Assist MVP

Experimental assistive-vision prototype: a YOLOv8n obstacle detector trained on the ROD
dataset, a rule-based alert prioritiser, and offline spoken alerts, packaged as a
reproducible image-to-video demo.

> **Not a navigation aid.** This prototype processes still images. It has no live camera
> support, does not estimate distance, does not give directions, and has not been
> evaluated for safety. Do not rely on it to move around.

## What it does

```
images ──▶ detector.py ──▶ guidance.py ──────────▶ speech.py ──▶ demo.py
           YOLOv8n          tier + image region     eSpeak NG      frames, WAV, MP4,
           (existing        top-2 per frame,        offline TTS    transcript, JSON
            checkpoint)     6 s repeat cooldown                    with latency
```

| Module | Responsibility |
|---|---|
| `vision_assist/detector.py` | Loads the existing checkpoint, returns plain `Detection` objects plus an annotated frame |
| `vision_assist/guidance.py` | Pure rules: hazard tier per class, image third (left/centre/right), ranking, wording, repeat suppression |
| `vision_assist/speech.py` | eSpeak NG synthesis with timeouts and errors, plus WAV timeline assembly |
| `vision_assist/demo.py` | CLI and `run_demo()`: writes all outputs and per-stage latency |
| `vision_assist/config.py` | Path resolution: CLI flag, then environment variable, then default (no Kaggle-only paths) |

### Alert rules (`guidance.py`)
- **High tier:** stairs, manhole, car, motorcycle, bike, truck, bus. These are prefixed with "Caution."
- **Medium tier:** people, dogs, poles, cones, barrels, bins, benches, other street furniture.
- **Low tier:** crosswalk, traffic sign. **Ignored:** road, building (present in almost every frame).
- One alert per (class, image third). Order is tier first, then larger box, then confidence. At most **2** alerts are spoken per frame.
- The same (class, image third) is not repeated within **6 s**. A hazard that persists is re-announced after the cooldown.
- If nothing is detected, nothing is spoken. The system **never says "clear"**.
- "left/centre/right" is the third of the *image* that the box centre falls in. It is not a
  direction and assumes nothing about where the camera points. Box size is used only for
  ranking. It is **not** a distance estimate.

## Verified results

### Detector: validation split (recorded, not re-run in this PR)
From the executed training notebook `notebooks/vision-assist-v1.ipynb` and
`results/rod_yolov8n_50ep/metrics.json`. YOLOv8n at 640 px, 50 epochs, seed 42, ROD
validation split of 3,511 images and 5,572 instances.

| Metric | Value |
|---|---|
| mAP50 | **0.907** |
| mAP50-95 | **0.726** |
| Precision | 0.874 |
| Recall | 0.841 |

Selected high-tier classes (mAP50): stairs 0.943, manhole 0.971, car 0.888, bus 0.921,
motorcycle 0.794. Person: 0.847.

**Not yet measured:** test-split metrics for this checkpoint. The earlier figures in this
README (mAP50 0.6746) were from a one-epoch smoke run (`models/rod_baseline_smoke_best.pt`)
and are superseded.

### Pipeline: tests and latency (run on 2026-10-13)
- `pytest`: **42 passed**, 0 failed, 0 skipped. Run on Linux (WSL2) with CPU-only PyTorch,
  Python 3.12, ultralytics 8.4.175, espeak-ng 1.52, and ffmpeg from `imageio-ffmpeg`.
- CPU inference at 640 px, median of 10 runs on `bus.jpg`: **~140 ms**. The demo run reports
  ~180 ms mean over 2 frames, ~15–22 ms per TTS clip and ~235 ms from detection to audio file
  per frame. These numbers are from one laptop CPU, are not a real-time claim, and include no
  camera capture or audio playback.
- `notebooks/Vision_Assist_V2_Voice_Demo.ipynb` was executed top to bottom locally with
  nbconvert and produced no errors. Its Kaggle-specific branches (clone into `/kaggle/working`,
  apt install, ROD path auto-detection) have **not** been executed yet.

## Quick start

### Kaggle
1. Create a notebook from `notebooks/Vision_Assist_V2_Voice_Demo.ipynb`.
2. Use *Add Input* to attach `abtinzandi/obstacle-detection-dataset` (ROD), and enable Internet.
3. Run all cells. The notebook clones this repo, installs `requirements.txt` plus `espeak-ng`
   and `ffmpeg`, runs the tests, then the demo on the first 8 validation images.
   Set `BRANCH` in cell 1 if you are testing an unmerged branch.

### Local (Linux/macOS)
```bash
git clone https://github.com/SudhaMsr/VisionAssist-MVP.git && cd VisionAssist-MVP
python3.12 -m venv .venv && . .venv/bin/activate
# optional, CPU-only machines: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
sudo apt-get install -y espeak-ng        # macOS: brew install espeak-ng
pytest -q
python -m vision_assist.demo --images /path/to/images --out outputs/demo --limit 8
```
With no ROD images to hand, `--images "$(python -c 'import ultralytics,pathlib;print(pathlib.Path(ultralytics.__file__).parent/"assets")')"`
runs on the two sample photos bundled with Ultralytics. These photos are out-of-domain and only show that the pipeline works.

Configuration: `--model`/`VISION_ASSIST_MODEL`, `--images`/`VISION_ASSIST_IMAGES`,
`--out`/`VISION_ASSIST_OUTPUT`, `--tts`/`VISION_ASSIST_TTS`. See `python -m vision_assist.demo -h`
for thresholds, alert count and cooldown.

### Outputs (`outputs/demo/`, git-ignored)
| File | Contents |
|---|---|
| `frames/frame_NN.jpg` | Annotated image with a caption strip: what was spoken and what was suppressed |
| `vision_assist_demo.wav` | All alerts on one timeline. Each frame's slot is at least 2 s and stretches to fit its speech |
| `vision_assist_demo.mp4` | H.264 and AAC. Frames held for their slot duration, in sync with the WAV |
| `transcript.txt` | Timestamped spoken text |
| `demo_results.json` | Settings, hardware, latency summary, per-frame detections and spoken/suppressed alerts |

## Known limitations
- **Missed large vehicle (observed):** on Ultralytics' `bus.jpg` (a bus seen side-on behind pedestrians, filling much of the frame) the
  checkpoint detects the people but **not the bus**, even at confidence 0.1. This is one
  out-of-domain image, not a measured miss rate, but it shows the model can fail on large, nearby
  vehicles. That is the case that matters most. The rule engine cannot recover a missed detection.
- Still images only. No camera capture, tracking, or temporal smoothing.
- No distance, depth, direction or path-clearance reasoning.
- The hazard tiers are hand-written heuristics. They have not been validated with users.
- Test-split accuracy has not been measured for the 50-epoch checkpoint. Performance outside
  ROD-like imagery is unknown.
- The checkpoint's class names include a dataset typo (`Teraffic Barrel`). It is kept as-is so
  the labels match the weights, and it is spoken as "traffic barrel".

## Repository layout
```
vision_assist/        reusable package (detector, guidance, speech, demo CLI)
tests/                pytest suite (fake TTS engine + real-model smoke tests)
notebooks/
  vision-assist-v1.ipynb              executed training + validation run (source of metrics)
  vision-assist-v1 (3).ipynb          exploratory scratch copy (mostly unexecuted; kept for history)
  Vision_Assist_V2_Voice_Demo.ipynb   thin runner around vision_assist (Kaggle or local)
models/               rod_yolov8n_50ep_best.pt (used), rod_baseline_smoke_best.pt (1-epoch smoke)
results/
  rod_yolov8n_50ep/   validation metrics, curves, confusion matrices
  voice_demo/         LEGACY output of an earlier ad-hoc notebook cell (pre-rule-engine);
                      not reproducible from current code
configs/rod_detection.yaml  class names + dataset layout used for training (Kaggle paths;
                      only the class list is used by tests)
```

Dataset images are not included. ROD is obtained separately from Kaggle.

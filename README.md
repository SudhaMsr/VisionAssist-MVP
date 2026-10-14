# Vision Assist MVP

Experimental assistive-vision prototype: a YOLOv8n obstacle detector trained on the ROD
dataset, a rule-based alert prioritiser, and offline spoken alerts, packaged as a
reproducible image-to-video demo.

> **Not a navigation aid.** This prototype processes still images. It has no live camera
> support, does not estimate distance, does not give directions, and has not been
> evaluated for safety or tested with users. It can miss or misclassify obstacles, and
> silence does not mean the way is clear. Do not rely on it to move around, and never use it
> as the sole means of navigation.

## Demo

![Annotated demo frames with the spoken alert under each image](results/demo/preview.gif)

**▶ Video with speech: [`results/demo/vision_assist_demo.mp4`](results/demo/vision_assist_demo.mp4)**
(29.8 s). Also in that folder: the audio track ([`vision_assist_demo.wav`](results/demo/vision_assist_demo.wav)),
the [transcript](results/demo/transcript.txt) and the per-frame details
([`demo_results.json`](results/demo/demo_results.json)). GitHub shows the GIF inline; to hear
the audio, download the MP4 from its page.

The video was generated locally on CPU on 2026-10-14 by `python -m vision_assist.demo`,
using 8 real ROD **validation** images listed in [`configs/demo_images.txt`](configs/demo_images.txt):
six scenes containing several hazards, plus one stairs image and one manhole image. They were
chosen by `scripts/select_demo_images.py` from **ground-truth labels only**, never from model
output, so the demo shows the model's misses as well as its hits. Comparing labels with
detections across the 8 images: 5 labelled hazards were missed, and there were no false-class detections.
It replaces an earlier Kaggle run (2026-10-13) of `notebooks/Vision_Assist_V2_Voice_Demo.ipynb`. The detections
are identical. Only frames 1 and 6 are spoken differently, because of the ranking change described under
Alert rules.

| # | Spoken alert | Notes |
|---|---|---|
| 1 | "Caution. Car, centre. Person, left." | The labelled **motorcycle was missed**. A tree and a pole were detected but not spoken (2-alert limit) |
| 2 | "Caution. Bus, centre." | The labelled **truck was missed** |
| 3 | "Caution. Motorcycle, right. Tree, centre." | |
| 4 | "Caution. Motorcycle, left." | |
| 5 | "Person, right. Crosswalk, centre." | The labelled **car was missed** |
| 6 | "Person, centre. Tree, centre." | The labelled **car and motorcycle were missed**. People are ranked before trees of the same tier. "Person, right" was held back because frame 5 said it 3.9 s earlier |
| 7 | "Caution. Stairs, centre." | Detected at confidence 0.43, just above the 0.35 threshold |
| 8 | "Caution. Manhole, centre." | |

## Results at a glance

| | Value | Evidence |
|---|---|---|
| ROD **test** mAP50 / mAP50-95 (1,629 images) | **0.913 / 0.746** | `results/rod_yolov8n_50ep_test/metrics_test.json` |
| ROD test precision / recall | 0.902 / 0.845 | same |
| ROD validation mAP50 / mAP50-95 (3,511 images) | 0.907 / 0.726 | `results/rod_yolov8n_50ep/metrics.json` |
| Automated tests | 77 passed, 0 failed (locally with espeak-ng, 2026-10-14). The Kaggle runs predate the 32 newest tests | `pytest -q` |
| CPU inference per image (640 px) | Local CPU: mean 105 ms, median 96 ms (8 demo images, 2026-10-14). The earlier Kaggle CPU run gave a mean of 146 ms. Load affects it: about 45 ms per frame in the local stream runs | `results/demo/demo_results.json` → `latency` (local). The Kaggle figure is in that file's previous version in git |

The scores are object-detection accuracy on the ROD dataset. They do not measure how safe or
useful the spoken alerts are.

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
- One alert per (class, image third). Order is tier first. Within a tier, people and dogs come before static objects, then larger box, then confidence. At most **2** alerts are spoken per frame.
- The same (class, image third) is not repeated within **6 s**. A hazard that persists is re-announced after the cooldown.
- A lower-tier alert never takes the place of a higher-tier hazard that is being held back as a repeat. Without this rule, "Tree, right." was spoken alone while a motorcycle was in view (ROD test image IMG_20590).
- If nothing is detected, nothing is spoken. The system **never says "clear"**.
- "left/centre/right" is the third of the *image* that the box centre falls in. It is not a
  direction and assumes nothing about where the camera points. Box size is used only for
  ranking. It is **not** a distance estimate.

## Quick start

### Kaggle
1. Create a notebook from `notebooks/Vision_Assist_V2_Voice_Demo.ipynb`.
2. Use *Add Input* to attach `abtinzandi/obstacle-detection-dataset` (ROD), and enable Internet.
3. Run all cells. The notebook clones this repo (`BRANCH = "main"` by default), installs
   `requirements.txt` plus `espeak-ng` and `ffmpeg`, runs the tests, then builds the demo from
   `configs/demo_images.txt` into `outputs/demo/`. Set `IMAGE_LIST = None` to use the first 8
   validation images instead.

### Local (Linux/macOS)
```bash
git clone https://github.com/SudhaMsr/VisionAssist-MVP.git && cd VisionAssist-MVP
python3.12 -m venv .venv && . .venv/bin/activate
# optional, CPU-only machines: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
sudo apt-get install -y espeak-ng        # macOS: brew install espeak-ng
pytest -q
# Download ROD from kaggle.com/datasets/abtinzandi/obstacle-detection-dataset and unzip it into
# datasets/ so that datasets/ROD-Dataset/dataset/valid/images exists (datasets/ is git-ignored). Then:
python -m vision_assist.demo
```
With no arguments, the demo uses the ROD validation images in `datasets/` (or on Kaggle) and the 8
images listed in `configs/demo_images.txt`. This is the same demo as above, written to `outputs/demo/`.
If ROD is somewhere else, pass `--images /path/to/ROD-Dataset/dataset/valid/images`.
`--limit N` uses the first N images in sorted order instead. To hear the alerts, open
`outputs/demo/vision_assist_demo.mp4` in any video player. To see or hear the demo without
running anything, open the committed copy in `results/demo/`.

The notebook also runs locally, with no cells to edit: `pip install notebook`, run
`jupyter notebook notebooks/Vision_Assist_V2_Voice_Demo.ipynb` from the repo, then run all cells.
Without root, install espeak-ng yourself first. If it is not on `PATH`, set `VISION_ASSIST_TTS`
to the executable.

Without ROD, `--images "$(python -c 'import ultralytics,pathlib;print(pathlib.Path(ultralytics.__file__).parent/"assets")')"`
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

Each run first deletes the files listed above from an earlier run in the same folder. Other files
are not touched. Old and new outputs therefore never mix, even if a run fails part-way. Before anything is written, the demo
checks for a missing or corrupt checkpoint, unreadable or unsupported inputs (such as video files),
missing espeak-ng and invalid settings. If any check fails, it prints a one-line `error:` message,
exits with code 1 and writes nothing. An output folder that cannot be written, or a speech failure
during the run, is also reported as a one-line `error:`.

## Stream timing (experimental)
The slideshow gives every image as long as its speech needs, so alerts can never collide.
`vision_assist.stream` instead plays frames at a fixed rate, and speech takes real time:
```bash
python -m vision_assist.stream --image-list configs/stream_people_clip.txt --fps 1   # or --video FILE
```
- There is **one speech channel**. While a message plays, plus 0.5 s, nothing new is spoken
  and nothing is queued. When the channel is free, the *current* frame is evaluated, so a
  hazard that has gone is never announced late. The alert rules are the same as the slideshow's.
- **Urgent alerts interrupt.** If the unchanged rules would speak a high-tier alert (stairs,
  manhole or a vehicle) while a lower-tier message is playing, that message is cut off,
  with a 20 ms fade, and the alert is spoken at once. High-tier speech is never interrupted.
  `--no-interrupt` restores the earlier behaviour.
- Outputs go to `outputs/stream/`: frames whose caption shows what was held back or cut off,
  the WAV/MP4 timeline (cut messages really are cut), `transcript.txt` (`[cut off]` marks a cut)
  and `stream_results.json` (metrics plus per-frame detail).
- The clips are ROD frames that Roboflow extracted from two videos, listed in order by
  `scripts/select_stream_clips.py`. Some frames are from the training split, so the clips test
  timing, not accuracy. The original frame rate is unknown, so results are given at 1 and 2 fps.

Measured on 2026-10-14 (CPU, espeak-ng 1.52). "Before" is `--no-interrupt`; "after" is the default:

| Clip | fps | Car warnings announced, before → after | Car warning latency (after) | Messages/min | Speaking | Overlaps | Messages cut off |
|---|---|---|---|---|---|---|---|
| People (119 consecutive frames) | 1 | 0 → **1 of 1** | 0 s | 16.6 → 16.6 | 66% → 68% | 0 | 0 |
| People | 2 | 0 → **1 of 1** | 0 s | 18.2 → 19.2 | 74% → 75% | 0 | 1 |
| Dashcam (135 frames with gaps) | 1 | 4 → **7 of 8** | 0 s | 13.3 → 14.2 | 56% → 58% | 0 | 3 |
| Dashcam | 2 | 0 → **7 of 8** | 0 s | 18.7 → 19.6 | 60% → 73% | 0 | 3 |

Before this mode, the same people clip at 1 fps had 21 alerts that would have started over
earlier speech. Speech no longer overlaps. Remaining issues:
- **One car warning per dashcam run still waits.** It arrived while another car warning was
  playing (high-tier speech is not interrupted), and the car had gone by the time that warning
  finished.
- **Still talkative.** The channel is busy 58–75% of the time, because a person who stays in
  view is re-announced every 6 s and again whenever they cross into another image third. On the
  dashcam clip at 2 fps, the rise from 60% to 73% is the 7 car warnings that are no longer lost.
- The latencies assume speech starts at the frame's timestamp. In a live system, inference
  (about 45 ms) and TTS (about 8 ms) would be added. The person and car detections were not
  checked by hand.

Following the same object across frames (not implemented yet) is the next step for the
talkativeness.

## Known limitations
- **Still images only.** No camera capture, tracking or temporal smoothing. No distance,
  depth, direction or path-clearance reasoning. Not safety-validated and not tested with users.
- **The detector misses objects.** In the demo it missed 5 labelled hazards across 4 of the 8 images:
  a motorcycle (frame 1), a truck (frame 2), a car (frame 5), and a car and motorcycle (frame 6). On Ultralytics' out-of-domain `bus.jpg` it missed a large bus even at
  confidence 0.1. The rule engine cannot recover a missed detection.
- **Ranking uses class and box size, not distance.** Within a tier, people and dogs now rank
  before static objects. This fixed demo frame 6, where two tree canopies took both slots from five
  pedestrians. On all 3,511 ROD validation images, a person or dog was displaced by a static
  object 11 times before the change and 0 times after it. The trade-off is that a close static
  obstacle can go unsaid behind distant people. This happened once in 3,511 images: in IMG_22440,
  a bench in the centre covering 30% of the image is dropped for two distant people. A parked
  motorcycle still outranks a person walking toward the camera (ROD test IMG_20595). Class stands
  in for movement: the system cannot tell whether an object is actually moving. The tiers are
  hand-written heuristics.
- **Accuracy is known only on ROD.** Performance on other imagery, such as a phone camera at
  walking height, has not been measured.
- The checkpoint's class names include a dataset typo (`Teraffic Barrel`). It is kept as-is so
  the labels match the weights, and it is spoken as "traffic barrel".

## Verification details
- **Detector:** YOLOv8n at 640 px, 50 epochs, seed 42. Test metrics were measured on Kaggle with
  `scripts/evaluate_checkpoint.py`, using the same label preparation as training (1,957 boxes and
  415 polygons converted, 0 skipped). Test mAP50 for the high-tier classes: stairs 0.888,
  manhole 0.950, car 0.896, bus 0.881, motorcycle 0.880, truck 0.885, bike 0.963. Person scores
  0.873. The weakest classes are road (0.718, ignored by the alert rules) and dustbin (0.781).
  Test agrees with validation, so there is no evidence on ROD that retraining is needed.
- **Class names:** the dataset's own `data.yaml`, the checkpoint and `configs/rod_detection.yaml`
  list the same 25 classes in the same order.
- **Notebook on Kaggle (2026-10-13):**
  - From `main` with the default `BRANCH = "main"`: completed, 43 of 43 tests passed (before this
    milestone's 2 new tests), and produced a 20.2 s MP4 on the first 8 validation images.
    A record of that earlier run is in `results/voice_demo_v2_kaggle/`.
  - With the curated image list: completed, 45 of 45 tests passed, and produced the first
    version of `results/demo/`. In both runs the MP4 decodes with no errors, and speech is
    present in exactly the slots that have an alert.
- **Local demo run (2026-10-14, CPU, WSL2):** `results/demo/` was regenerated after people and
  dogs were ranked ahead of static objects. The detections are identical to the Kaggle run, and
  only frames 1 and 6 are spoken differently. The MP4 decodes with no errors. Audio, video and
  frame slots each last 29.8 s, and speech is present in exactly the slots that have an alert.
  77 of 77 tests pass locally. The Kaggle notebook has not been re-run since this change.
- **Multi-hazard check on ROD test images** (IMG_20581–20597): this run found the
  tier-replacement flaw now handled by the alert rules. Its record is in
  `results/voice_demo_v2_rod_multi_hazard/`.
- **Local runs:** WSL2 Linux, Python 3.12, CPU-only PyTorch, ultralytics 8.4.175, espeak-ng 1.52.
  Latency figures come from single CPUs, are not real-time claims, and exclude camera capture and
  audio playback.
- **Checkpoint:** `models/rod_yolov8n_50ep_best.pt` (6.5 MB, committed directly, no Git LFS needed;
  SHA-256 `6f5647e9…968c8a`). It is the default for `--model`. A copy recovered on 2026-10-14 was
  byte-identical to it.
- **Recovered legacy export (2026-10-14):** `archive/results/voice_demo_v2_recovered/` is an older
  export that uses the pre-`vision_assist` wording ("Caution. Car detected."). Its MP4 is cut off
  at 16 s and its audio drifts out of step with the frames, so the last alerts are never heard.
  It is kept for provenance only. See `archive/README.md`. The current demo is `results/demo/`.
- The old README figure (mAP50 0.6746) came from a one-epoch smoke run
  (`models/rod_baseline_smoke_best.pt`) and is superseded.

## Repository layout
```
vision_assist/        reusable package (detector, guidance, speech, demo CLI, stream timing CLI)
tests/                pytest suite (fake TTS engine + real-model smoke tests)
scripts/
  evaluate_checkpoint.py   test/valid-split evaluation (same label prep as training)
  select_demo_images.py    regenerates configs/demo_images.txt from ROD labels
  select_stream_clips.py   regenerates the two configs/stream_*_clip.txt frame lists
configs/
  demo_images.txt          the 8 demo images, in playback order
  stream_*_clip.txt        ordered ROD video frames for stream timing (people, dashcam)
  rod_detection.yaml       class names + dataset layout used for training
notebooks/
  vision-assist-v1.ipynb              executed training + validation run (source of metrics)
  Vision_Assist_V2_Voice_Demo.ipynb   demo runner (Kaggle or local)
models/               rod_yolov8n_50ep_best.pt (used), rod_baseline_smoke_best.pt (1-epoch smoke)
results/
  demo/                            the current demo (MP4, WAV, GIF, transcript, JSON)
  rod_yolov8n_50ep/                validation metrics, curves, confusion matrices
  rod_yolov8n_50ep_test/           test-split metrics
  voice_demo_v2_kaggle/            earlier Kaggle run (first 8 validation images)
  voice_demo_v2_rod_multi_hazard/  multi-hazard test-image check
archive/              superseded scratch notebook, legacy demo and recovered legacy export (see archive/README.md)
```

Dataset images are not included. ROD is obtained separately from Kaggle.

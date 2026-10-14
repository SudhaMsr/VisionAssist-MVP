# Fine-tune v1: plan and acceptance criteria

Status: **pipeline prepared, training not started.** This documents what
`notebooks/finetune_v1.ipynb` and `training/` do, the decisions already made, and the
decisions still needed before a training run.

## Goal

Fine-tune the existing `models/rod_yolov8n_50ep_best.pt` (unchanged, never overwritten)
so a new checkpoint:

1. detects cars and people much better on real phone footage, beating today's model by
   at least 15 recall points and at least matching the COCO-assist experiment
   (`vision_assist/detector.py`'s `assist_model_path`), at the app's 0.35 confidence
   threshold;
2. keeps every ROD class's accuracy within 0.02 mAP50 of today's model on a **clean**
   ROD test set, with stairs, manholes and traffic cones within 0.01;
3. keeps the app's alert-safety guarantees: 0 fall hazards pushed out of the 2 alert
   slots, people pushed out by static objects no more often than today, 0 speech
   overlaps on the existing stream clips.

All 25 ROD classes and their IDs are unchanged — a fine-tuned checkpoint is a drop-in
replacement for `models/rod_yolov8n_50ep_best.pt` with no code changes, if and when it is
approved.

## Decisions already made

- **ROD "Bike" labels are left unchanged** (not split into bicycle/motorbike). COCO
  images containing a bicycle are excluded from the training subset instead
  (`training/coco_subset.py`), so no image teaches the model an inconsistent "Bike"
  boundary. COCO's `bicycle` category is not mapped to any ROD class.
- **Re-split ROD by source, not Roboflow's own split**: about 21% of ROD's own
  validation/test images share a source photo or video with training (duplicates and
  video frames), which makes today's reported test scores optimistic. See
  `training/splits.py` and the measurement below.
- **COCO subset is street-scene only**, capped at 35% of training images
  (`--coco-max-frac`), so ROD's hazard classes stay well represented.
- **Missing labels are filled in both directions** before training
  (`training/label_completion.py`), with a mandatory human review step, because both
  source datasets are only partially labelled for the other's classes.

## Decisions still needed from you

1. **Phone footage must be recorded and labelled.** See `docs/phone_footage_spec.md` for
   exactly what and how. Without it, run A trains on ROD+COCO only and the real-world
   acceptance checks (criterion 1 above) cannot be evaluated — `training/evaluate.py`
   reports them as `MISSING`, not as a pass.
2. **Licensing, before anything is published**: Ultralytics is AGPL-3.0; the repository
   has no LICENSE file yet. ROD is MIT on Kaggle but aggregates 26 Roboflow sources with
   their own licences (see the dataset's paper reference). COCO images carry individual
   Flickr licences — fine for training, not for redistributing the images themselves.
3. **When to actually train.** The notebook's training cell is guarded by
   `RUN_TRAINING = False`; nothing trains until that is flipped and the cell is re-run,
   and even then only on Kaggle, never locally.

## Measured: why the re-split matters

Run on the currently attached ROD copy (`training/splits.py`, seed 42, grouped by source
before Roboflow's `.rf.` suffix, or by source video for video-frame images):

- 16,031 distinct source groups found.
- Re-split result: ~19,460 train / ~2,433 val / ~2,433 test images, **0 groups shared
  between any two splits** (verified by the script itself; it raises if any are found).
- For comparison, ROD's own split has about 21% of validation and 21% of test images
  sharing a source with training.

The committed split lists are `configs/finetune_v1/splits/rod_{train,val,test}.txt`
(paths relative to `ROD-Dataset/dataset`) and `rod_split_report.json` (per-class counts
per split, for checking class balance).

## Pipeline (what each script does)

| Step | Script | Input -> Output |
|---|---|---|
| 1. Re-split ROD | `training/splits.py` | ROD's `rename_map.csv` -> the committed split lists above |
| 2. Materialise ROD | `training/build_dataset.py prepare-rod` | split lists + attached ROD -> `<work>/rod_{train,val,test}/{images,labels}` (symlinked images, converted labels) |
| 3. COCO subset | `training/coco_subset.py` | attached COCO 2017 -> `<work>/coco_{train,val}/{images,labels}` (street-scene images only, ROD-format labels) |
| 4. Phone frames | `training/phone_frames.py` | your walk videos -> one folder per walk for labelling (see the spec doc) |
| 5. Label completion | `training/label_completion.py` | fills in ROD-only classes in COCO images (via today's ROD model) and people/vehicles in ROD images (via the COCO model); writes a review sheet and a full log of every box added |
| 6. Assemble | `training/build_dataset.py assemble` | combines ROD train + capped COCO + phone train/val walks -> `data_v1.yaml` (run A) and `data_v1_rod_only.yaml` (run B control); test sets are written separately and never trained on |
| 7. Train (guarded) | `notebooks/finetune_v1.ipynb`, settings in `configs/finetune_v1/train.yaml` | starts from the shipped checkpoint; run A (ROD+COCO[+phone]) and run B (ROD only, same schedule) |
| 8. Evaluate | `training/evaluate.py` | compares all systems on the held-out test sets and checks the criteria above |

## Hyperparameters (`configs/finetune_v1/train.yaml`)

Both runs start from `models/rod_yolov8n_50ep_best.pt` (never from the plain COCO
`yolov8n.pt`, and never overwriting it). Key differences from the original 50-epoch run,
chosen for a lower-risk fine-tune rather than training from scratch:

| Setting | Original run | Fine-tune v1 | Why |
|---|---|---|---|
| epochs / patience | 50 / 10 | 30 / 10 | fine-tune, not from scratch |
| optimizer | `auto` -> picked `MuSGD` | `SGD` (pinned) | reproducible; avoids Ultralytics' automatic choice changing between runs |
| lr0 | 0.01 | 0.002 | 5x lower, to adjust rather than overwrite existing weights |
| batch | 8 | 32 | Kaggle T4/P100 has headroom; same `imgsz` 640 |
| everything else | — | unchanged (seed 42, deterministic, AMP, augmentation) | keep as much as possible identical to the original recipe |

Estimated GPU time: roughly 1.5-2 hours per run on a T4 (the original 50-epoch run with
batch 8 took 2.9 hours; this has ~1.6x the images at 0.6x the epochs and a larger batch).
Both runs fit comfortably in Kaggle's weekly free GPU quota.

## Evaluation (`training/evaluate.py`)

Every system (today's model; today's model + COCO assist; run A; run B) is evaluated the
same way, through `vision_assist.detector.Detector` so the app's actual merge logic is
exercised:

- **App-threshold recall/precision** (conf 0.35, IoU 0.5) per class, on the clean ROD
  test set, the phone test walks, and a COCO check set (for the 10 classes COCO and ROD
  share).
- **Ultralytics mAP50 / mAP50-95** per class, for single models (not meaningful for the
  two-model assist, since it is not one Ultralytics model).
- **Alert simulation**: the same `vision_assist.guidance.prioritise` rules used in the
  app, run on the test-set detections, counting fall-hazard displacement, lost high-tier
  alerts, and people displaced by static objects — the same checks used when validating
  the COCO-assist and fall-hazard-ranking changes earlier in this project.
- **Median inference time** per system.

Results are written as `eval_report.json` (machine-readable) and `eval_report.md` (a
table per test set, plus a PASS/FAIL line per acceptance criterion per candidate).

## Git workflow

1. This branch, `experiment/finetune-v1`, holds only code: the `training/` package, the
   notebook, the committed split lists, and this documentation. **No dataset, no
   checkpoint, no training run is committed here.**
2. Run the notebook on Kaggle with ROD and COCO attached (and, once ready, your private
   phone-footage dataset). Review the label-completion sheets before training, and leave
   `RUN_TRAINING = False` until you decide to proceed.
3. Download the notebook's Output folder from the Kaggle website (no local Kaggle API
   credentials are configured yet, so the CLI path is not available). Verify the
   checksum Kaggle reports against the one `ultralytics` prints after training.
4. Locally: copy the chosen checkpoint to `models/rod_coco_ft_v1_best.pt` (new file;
   `rod_yolov8n_50ep_best.pt` is untouched) and the metrics to
   `results/rod_coco_ft_v1/`. Commit these on this branch and open a PR with the
   `eval_report.md` comparison table.
5. **Switching the app's default model is a separate PR**, made only after you review
   the comparison and explicitly approve it — exactly as with the COCO-assist and
   fall-hazard changes earlier in this project. That PR would also regenerate
   `results/demo/` with your approval, as before.

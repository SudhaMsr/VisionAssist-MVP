# Phone footage needed for fine-tune v1

This is the missing input for the plan in `docs/finetune_v1_plan.md`. Without it there is
no way to measure a real-world improvement, and the acceptance checks for car/person
recall cannot run (`training/evaluate.py` reports them as `MISSING` rather than passing
silently).

## What to record

**8 separate walks, each 1-3 minutes, in different streets and lighting.** Phone in
landscape, held at chest height, pointing forward along your path of travel — the same
way you tested the app already (`outputs/ui_runs/...VID_20261009_101055` is a good
example of the framing to repeat). Vary:

- street type: a quiet residential street, a busier road with parked and moving traffic,
  a path with a crosswalk, somewhere with a bin/pole/bench if convenient
- lighting: at least one walk in each of bright, overcast and low light
- at least one walk should contain a clear approaching vehicle and a clear pedestrian
  crossing your path, since these are the cases today's model misses

Save each walk as its own `.mp4`, named with only letters, digits, `-` and `_`
(`training/phone_frames.py` checks this), e.g.:

```
walk01_high_street_morning.mp4
walk02_park_road_overcast.mp4
walk03_crosswalk_evening.mp4
...
walk08_...
```

## Splitting into train / val / test (by walk, never by frame)

Assign whole walks — never individual frames of the same walk — to a split, in
`configs/finetune_v1/phone_walks.yaml`. Frames from the same walk are near-duplicates, so
splitting by frame would leak the test set into training.

| Split | Walks | Purpose |
|---|---|---|
| `train` | ~5 walks | goes into fine-tuning |
| `val` | ~1 walk | used during training to pick the best checkpoint |
| `test` | ~3 walks, **different streets from train/val** | never trained on; this is what `training/evaluate.py` reports recall on, and what the acceptance criteria are checked against |

## Extracting frames for labelling

```bash
python -m training.phone_frames --videos ~/walks --out ~/phone_dataset --fps 1
```

This writes one frame per second per walk (so consecutive frames are not
near-duplicates) to `~/phone_dataset/<walk_name>/images/`, and an empty
`~/phone_dataset/<walk_name>/labels/` folder to fill in. One second of video is about
1,200 total frames across 8 walks; at a steady labelling pace that is the main time cost
of this step.

## Labelling

Any tool that exports **YOLO format** (one `.txt` per image, `class cx cy w h`,
normalised 0-1, one line per object) works — CVAT and Label Studio are both free. Import
`<walk>/images/`, export back into `<walk>/labels/` with matching file names.

- Use all 25 classes from `configs/rod_detection.yaml` where relevant: not just Car and
  Person, but Stairs, Manhole, Traffic Cone etc. whenever they appear, so the test set
  also measures whether ROD-specific hazards are preserved.
- Leave ROD's existing "Bike" mixing (bicycles and motorbikes both labelled "Bike")
  as-is; do not introduce a separate bicycle class.
- An image with nothing to label gets an empty (zero-byte) `.txt` file, not a missing
  one — `training/build_dataset.py` treats a missing label file as an error, to catch
  unlabelled images rather than silently skipping them.

## Privacy

This footage will contain bystanders' faces and vehicle plates.

- **Keep it private.** Upload it to Kaggle only as a **Private** dataset (never commit it
  to this repository — `datasets/` and `outputs/` are already git-ignored for this
  reason).
- **Blur faces and plates** in anything you share outside your own Kaggle account or this
  pipeline, e.g. demo screenshots or review sheets you show someone else.
- Record on streets, not on private property, and avoid lingering on identifiable people.

## Uploading to Kaggle

Zip `phone_dataset/` (one folder per walk, each with `images/` and `labels/`) and create
a **Private** Kaggle dataset from it. In `notebooks/finetune_v1.ipynb`, set:

```python
PHONE_ROOT = Path("/kaggle/input/<your-private-dataset-slug>")
```

Without `PHONE_ROOT` set, the pipeline still runs (ROD + COCO only); the notebook prints
which checks were skipped for lack of phone data.

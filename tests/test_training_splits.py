"""training.splits: ROD re-split by source group, no leakage, deterministic."""

import csv

import pytest

from training.splits import assign, class_counts, group_images, source_group


def rows(*specs):
    """specs: (split, old_stem, new_stem). image_ext defaults to .jpg."""
    return [{"split": s, "old_stem": o, "new_stem": n, "image_ext": ".jpg"} for s, o, n in specs]


def test_source_group_strips_roboflow_suffix_and_groups_duplicates():
    assert source_group("photo123.rf.abcdef0123456789") == "photo123"
    assert source_group("photo123_jpg.rf.deadbeef") == "photo123_jpg"


def test_source_group_collapses_video_frames_to_one_group():
    assert source_group("walk_mp4-0_jpg.rf.aaa") == "walk_mp4"
    assert source_group("walk_mp4-118_jpg.rf.bbb") == "walk_mp4"


def test_group_images_builds_split_aware_paths():
    g = group_images(rows(("train", "a.rf.111", "IMG_001"), ("valid", "a.rf.222", "IMG_900")))
    assert g["a"] == ["train/images/IMG_001.jpg", "valid/images/IMG_900.jpg"]


def test_assign_never_splits_a_source_group_and_is_deterministic():
    data = rows(*[("train", f"img{i}.rf.x", f"IMG_{i:03d}") for i in range(100)])
    groups = group_images(data)
    a = assign(groups, seed=42)
    b = assign(groups, seed=42)
    assert a == b
    seen = {img: split for split, imgs in a.items() for img in imgs}
    assert len(seen) == sum(len(v) for v in a.values())  # no image in two splits
    assert set().union(*a.values()) == set().union(*groups.values())  # nothing lost


def test_assign_keeps_a_multi_copy_source_together():
    groups = group_images(rows(("train", "shared.rf.1", "IMG_1"), ("train", "shared.rf.2", "IMG_2"),
                               *[("train", f"x{i}.rf.1", f"IMG_{i}") for i in range(20)]))
    splits = assign(groups, seed=1)
    homes = {s for s, imgs in splits.items() if any(i in imgs for i in groups["shared"])}
    assert len(homes) == 1  # both copies of "shared" landed in the same split


def test_assign_raises_on_bad_fractions():
    with pytest.raises(ValueError):
        assign({"a": ["x"]}, test_frac=0.6, val_frac=0.6)


def test_class_counts_uses_training_label_conversion(tmp_path):
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "a.txt").write_text("3 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.2\n")
    counts = class_counts(tmp_path, ["train/images/a.jpg"])
    from training.common import CLASS_NAMES

    assert counts[CLASS_NAMES[3]] == 1 and counts[CLASS_NAMES[0]] == 1  # bbox + converted polygon
    assert all(v == 0 for c, v in counts.items() if c not in (CLASS_NAMES[3], CLASS_NAMES[0]))


def test_main_writes_split_lists_with_no_leakage(tmp_path):
    from training.splits import main

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data = rows(*[("train", f"img{i}.rf.x", f"IMG_{i:04d}") for i in range(300)])
    with (dataset / "rename_map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "old_stem", "new_stem", "image_ext"])
        w.writeheader()
        w.writerows(data)
    for row in data:
        d = dataset / row["split"] / "labels"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{row['new_stem']}.txt").touch()

    out = tmp_path / "out"
    assert main(["--dataset", str(dataset), "--out", str(out), "--seed", "7"]) == 0
    files = {s: set((out / f"rod_{s}.txt").read_text().splitlines()) for s in ("train", "val", "test")}
    files = {s: {ln for ln in v if not ln.startswith("#")} for s, v in files.items()}
    assert not (files["train"] & files["val"]) and not (files["train"] & files["test"])
    assert len(files["val"]) and len(files["test"])  # non-empty with 300 images at default fractions
    import json

    report = json.loads((out / "rod_split_report.json").read_text())
    assert all(v == 0 for v in report["shared_groups_between_splits"].values())

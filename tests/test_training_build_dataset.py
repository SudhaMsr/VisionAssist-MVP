"""training.build_dataset: materialising ROD splits and assembling the training lists."""

import pytest

from training.build_dataset import cap_coco, phone_images, prepare_rod


def make_rod(tmp_path, split, names_and_labels):
    images, labels = tmp_path / "rod" / split / "images", tmp_path / "rod" / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name, label in names_and_labels:
        (images / f"{name}.jpg").write_bytes(b"x")
        (labels / f"{name}.txt").write_text(label)
    return tmp_path / "rod"


def test_prepare_rod_symlinks_images_and_converts_labels(tmp_path):
    rod = make_rod(tmp_path, "train", [("a", "2 0.5 0.5 0.2 0.2\n")])
    make_rod(tmp_path, "val", [])
    make_rod(tmp_path, "test", [])
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "rod_train.txt").write_text("train/images/a.jpg\n")
    (split_dir / "rod_val.txt").write_text("")
    (split_dir / "rod_test.txt").write_text("")
    work = tmp_path / "work"

    report = prepare_rod(rod, work, split_dir)
    assert report["train"]["images"] == 1 and report["train"]["bbox"] == 1
    linked = work / "rod_train" / "images" / "a.jpg"
    assert linked.is_symlink() and linked.resolve() == (rod / "train" / "images" / "a.jpg").resolve()
    assert (work / "rod_train" / "labels" / "a.txt").read_text().strip() == "2 0.500000 0.500000 0.200000 0.200000"


def test_prepare_rod_raises_on_missing_listed_image(tmp_path):
    rod = make_rod(tmp_path, "train", [])
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "rod_train.txt").write_text("train/images/missing.jpg\n")
    (split_dir / "rod_val.txt").write_text("")
    (split_dir / "rod_test.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        prepare_rod(rod, tmp_path / "work", split_dir)


def test_phone_images_requires_every_frame_labelled(tmp_path):
    walk = tmp_path / "phone" / "walk01" / "images"
    walk.mkdir(parents=True)
    (walk / "f1.jpg").write_bytes(b"x")
    (tmp_path / "phone" / "walk01" / "labels").mkdir()  # f1.txt missing
    walks_file = tmp_path / "walks.yaml"
    walks_file.write_text("train: [walk01]\nval: []\ntest: []\n")
    with pytest.raises(FileNotFoundError, match="no label file"):
        phone_images(tmp_path / "phone", walks_file)


def test_phone_images_rejects_a_walk_assigned_to_two_splits(tmp_path):
    walk = tmp_path / "phone" / "walk01" / "images"
    walk.mkdir(parents=True)
    walks_file = tmp_path / "walks.yaml"
    walks_file.write_text("train: [walk01]\nval: [walk01]\ntest: []\n")
    with pytest.raises(ValueError, match="walk01"):
        phone_images(tmp_path / "phone", walks_file)


def test_phone_images_none_root_returns_empty_without_touching_disk():
    assert phone_images(None) == {"train": [], "val": [], "test": []}


def test_cap_coco_keeps_everything_under_the_cap():
    coco = [f"img{i}" for i in range(10)]
    assert cap_coco(rod_train=100, phone_train=0, coco=coco, max_frac=0.5) == sorted(coco)


def test_cap_coco_samples_deterministically_when_over_the_cap():
    coco = [f"img{i}" for i in range(1000)]
    a = cap_coco(rod_train=100, phone_train=0, coco=coco, max_frac=0.35, seed=42)
    b = cap_coco(rod_train=100, phone_train=0, coco=coco, max_frac=0.35, seed=42)
    assert a == b
    share = len(a) / (100 + len(a))
    assert share <= 0.35 + 1e-9


def test_cap_coco_rejects_bad_fraction():
    with pytest.raises(ValueError):
        cap_coco(10, 0, ["a"], max_frac=1.0)

"""training.coco_subset: street-scene selection and label mapping (no real COCO download)."""

from training.coco_subset import labels_for, select


def coco_dict(images_anns):
    """images_anns: list of (image_id, width, height, [(category_name, bbox, iscrowd)])."""
    categories = sorted({name for _, _, _, anns in images_anns for name, _, _ in anns})
    cat_id = {n: i for i, n in enumerate(categories)}
    images, annotations = [], []
    for img_id, w, h, anns in images_anns:
        images.append({"id": img_id, "width": w, "height": h, "file_name": f"{img_id:012d}.jpg"})
        for name, bbox, crowd in anns:
            annotations.append({"image_id": img_id, "category_id": cat_id[name], "bbox": list(bbox),
                                "iscrowd": int(crowd)})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n} for n, i in cat_id.items()]}


def test_vehicle_images_are_selected():
    coco = coco_dict([(1, 640, 480, [("car", (0, 0, 10, 10), False)])])
    chosen, _, stats = select(coco, limit=None)
    assert [i["id"] for i in chosen] == [1] and stats["selected"] == 1


def test_person_alone_is_not_selected_but_person_with_outdoor_cue_is():
    coco = coco_dict([
        (1, 640, 480, [("person", (0, 0, 10, 10), False)]),
        (2, 640, 480, [("person", (0, 0, 10, 10), False), ("bench", (20, 20, 10, 10), False)]),
    ])
    chosen, _, stats = select(coco, limit=None)
    assert [i["id"] for i in chosen] == [2]
    assert stats["skipped: not street-like"] == 1


def test_bicycle_images_are_excluded():
    coco = coco_dict([(1, 640, 480, [("car", (0, 0, 10, 10), False), ("bicycle", (20, 20, 5, 5), False)])])
    chosen, _, stats = select(coco, limit=None)
    assert chosen == [] and stats["skipped: contains bicycle"] == 1


def test_crowd_images_are_excluded():
    coco = coco_dict([(1, 640, 480, [("car", (0, 0, 10, 10), False), ("person", (0, 0, 1, 1), True)])])
    chosen, _, stats = select(coco, limit=None)
    assert chosen == [] and stats["skipped: crowd region"] == 1


def test_limit_gives_a_deterministic_sample():
    coco = coco_dict([(i, 640, 480, [("car", (0, 0, 10, 10), False)]) for i in range(50)])
    a, _, _ = select(coco, limit=10, seed=42)
    b, _, _ = select(coco, limit=10, seed=42)
    assert [i["id"] for i in a] == [i["id"] for i in b] and len(a) == 10


def test_labels_for_maps_only_mapped_categories_and_converts_bbox():
    coco = coco_dict([(1, 100, 100, [("car", (10, 10, 20, 20), False), ("bicycle", (0, 0, 5, 5), False)])])
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns = [a for a in coco["annotations"] if a["image_id"] == 1]
    lines = labels_for(coco["images"][0], anns, cat_name)
    assert len(lines) == 1  # bicycle dropped, car kept
    from training.common import CLASS_ID

    assert int(lines[0].split()[0]) == CLASS_ID["Car"]

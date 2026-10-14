"""training.label_completion: merge logic for filling in missing labels (no real model calls)."""

from training.common import CLASS_ID
from training.label_completion import new_boxes


def test_predictions_added_when_no_covering_classes_given():
    # e.g. completing ROD-only classes on a COCO image: nothing existing to overlap.
    preds = [(CLASS_ID["Stairs"], 0.9, (0, 0, 10, 10)), (CLASS_ID["Manhole"], 0.6, (50, 50, 60, 60))]
    added = new_boxes([], preds, 100, 100, covering=None)
    assert [p[0] for p in added] == [CLASS_ID["Stairs"], CLASS_ID["Manhole"]]  # highest confidence first


def test_prediction_overlapping_a_covering_label_is_dropped():
    existing = [(CLASS_ID["Car"], 0.5, 0.5, 0.2, 0.2)]  # normalised row as read_yolo_labels returns
    preds = [(CLASS_ID["Car"], 0.8, (45, 45, 55, 55))]  # same car, in pixel xyxy at 100x100
    added = new_boxes(existing, preds, 100, 100, covering={CLASS_ID["Car"]})
    assert added == []


def test_prediction_not_overlapping_any_covering_label_is_kept():
    existing = [(CLASS_ID["Car"], 0.1, 0.1, 0.1, 0.1)]
    preds = [(CLASS_ID["Person"], 0.7, (80, 80, 90, 90))]
    added = new_boxes(existing, preds, 100, 100, covering={CLASS_ID["Car"], CLASS_ID["Person"]})
    assert len(added) == 1 and added[0][0] == CLASS_ID["Person"]


def test_covering_set_restricts_which_existing_labels_block_a_prediction():
    # A Stairs label should not block a Person prediction from being added (someone on the stairs).
    existing = [(CLASS_ID["Stairs"], 0.5, 0.5, 0.9, 0.9)]
    preds = [(CLASS_ID["Person"], 0.7, (40, 40, 60, 60))]
    added = new_boxes(existing, preds, 100, 100, covering={CLASS_ID["Person"]})
    assert len(added) == 1


def test_duplicate_predictions_among_themselves_keep_only_the_first():
    preds = [(CLASS_ID["Car"], 0.5, (10, 10, 20, 20)), (CLASS_ID["Car"], 0.9, (11, 11, 21, 21))]
    added = new_boxes([], preds, 100, 100, covering={CLASS_ID["Car"]})
    assert len(added) == 1 and added[0][1] == 0.9

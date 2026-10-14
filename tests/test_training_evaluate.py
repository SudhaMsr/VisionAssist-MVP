"""training.evaluate: system parsing, matching, acceptance checks (no real models)."""

import pytest

from training.evaluate import acceptance, compare_alerts, label_path, match, parse_system, rates


def test_parse_system_with_and_without_assist():
    assert parse_system("today=models/a.pt") == ("today", pytest_path("models/a.pt"), None)
    name, main, assist = parse_system("today+assist=models/a.pt+models/b.pt")
    assert (name, str(main), str(assist)) == ("today+assist", "models/a.pt", "models/b.pt")


def pytest_path(s):
    from pathlib import Path

    return Path(s)


def test_parse_system_rejects_malformed_spec():
    with pytest.raises(ValueError):
        parse_system("no-equals-sign")


def test_label_path_swaps_images_for_labels_anywhere_in_the_path():
    from pathlib import Path

    assert label_path(Path("/a/b/rod_test/images/sub/x.jpg")) == Path("/a/b/rod_test/labels/sub/x.txt")


def test_match_counts_tp_fp_fn_by_class():
    gt = [(0, (0, 0, 10, 10)), (1, (50, 50, 60, 60))]
    preds = [(0, 0.9, (0, 0, 10, 10)), (0, 0.5, (90, 90, 95, 95)), (2, 0.4, (50, 50, 60, 60))]
    c = match(gt, preds, classes=None)
    assert c["tp:" + name(0)] == 1 and c["fn:" + name(1)] == 1 and c["fp:" + name(0)] == 1 and c["fp:" + name(2)] == 1


def name(i):
    from training.common import CLASS_NAMES

    return CLASS_NAMES[i]


def test_match_respects_a_class_filter():
    gt = [(0, (0, 0, 10, 10)), (1, (50, 50, 60, 60))]
    preds = [(0, 0.9, (0, 0, 10, 10))]
    c = match(gt, preds, classes={name(0)})
    assert f"fn:{name(1)}" not in c and c[f"tp:{name(0)}"] == 1


def test_rates_computes_recall_and_precision():
    from collections import Counter

    c = Counter({f"tp:{name(0)}": 3, f"fp:{name(0)}": 1, f"fn:{name(0)}": 1})
    r = rates(c, [name(0)])[name(0)]
    assert r["recall"] == pytest.approx(0.75) and r["precision"] == pytest.approx(0.75) and r["labelled"] == 4


def test_rates_none_when_no_labelled_instances():
    from collections import Counter

    r = rates(Counter(), [name(0)])[name(0)]
    assert r["recall"] is None and r["precision"] is None and r["labelled"] == 0


def test_compare_alerts_detects_fall_hazard_and_high_tier_loss():
    from vision_assist.guidance import HIGH, Alert

    manhole = Alert("Manhole", HIGH, "centre", 0.9, 0.1)
    car = Alert("Car", HIGH, "left", 0.8, 0.3)
    ref = {"img1.jpg": [manhole, car]}
    other = {"img1.jpg": [car, Alert("Car", HIGH, "right", 0.6, 0.2)]}  # manhole pushed out
    c = compare_alerts(ref, other)
    assert c["fall_hazard_pushed_out"] == 1 and c["high_tier_alert_lost"] == 1 and c["messages_changed"] == 1


def test_compare_alerts_detects_person_displaced_by_static_object():
    from vision_assist.guidance import MEDIUM, Alert

    person = Alert("Person", MEDIUM, "centre", 0.8, 0.05)
    tree = Alert("Tree", MEDIUM, "left", 0.5, 0.3)
    ref = {"img1.jpg": [person]}
    other = {"img1.jpg": [tree]}
    c = compare_alerts(ref, other)
    assert c["person_or_dog_displaced_by_static"] == 1


def test_acceptance_reports_missing_when_no_phone_test_set():
    report = {"sets": {}, "alerts": {}}
    checks = acceptance(report, candidate="A", baseline="today", assist=None)
    assert "MISSING" in checks["phone test set"]


def test_acceptance_passes_when_recall_gain_and_precision_hold():
    report = {
        "sets": {
            "test_phone": {
                "systems": {
                    "today": {"app_threshold": {"Car": {"recall": 0.3, "precision": 0.8},
                                                 "Person": {"recall": 0.3, "precision": 0.8}}},
                    "A": {"app_threshold": {"Car": {"recall": 0.5, "precision": 0.78},
                                             "Person": {"recall": 0.5, "precision": 0.78}}},
                }
            }
        },
        "alerts": {"A": {"fall_hazard_pushed_out": 0, "person_or_dog_displaced_by_static": 0},
                   "_baseline_person_displaced": 0},
    }
    checks = acceptance(report, candidate="A", baseline="today", assist=None)
    assert all(v == "PASS" for v in checks.values())


def test_acceptance_fails_when_recall_gain_too_small():
    report = {
        "sets": {"test_phone": {"systems": {
            "today": {"app_threshold": {"Car": {"recall": 0.3, "precision": 0.8},
                                         "Person": {"recall": 0.3, "precision": 0.8}}},
            "A": {"app_threshold": {"Car": {"recall": 0.35, "precision": 0.8},
                                     "Person": {"recall": 0.3, "precision": 0.8}}},
        }}},
        "alerts": {},
    }
    checks = acceptance(report, candidate="A", baseline="today", assist=None)
    assert checks["phone Car recall +0.15 vs today"] == "FAIL"

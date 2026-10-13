from pathlib import Path

import pytest
import yaml

from vision_assist.detector import Detection
from vision_assist.guidance import (
    HAZARD_TIERS,
    HIGH,
    IGNORED_CLASSES,
    AlertScheduler,
    compose_message,
    image_region,
    prioritise,
)

REPO = Path(__file__).resolve().parent.parent


def det(name, x1=100, x2=200, y1=100, y2=200, conf=0.9, w=600, h=400):
    return Detection(name, conf, (x1, y1, x2, y2), w, h)


def test_every_dataset_class_has_a_rule():
    names = set(yaml.safe_load((REPO / "configs" / "rod_detection.yaml").read_text())["names"])
    covered = set(HAZARD_TIERS) | IGNORED_CLASSES
    assert names == covered, f"missing: {names - covered}, unknown: {covered - names}"


@pytest.mark.parametrize("x1,x2,region", [(0, 100, "left"), (250, 350, "centre"), (500, 600, "right")])
def test_image_region_uses_box_centre(x1, x2, region):
    assert image_region(det("Car", x1=x1, x2=x2)) == region


def test_high_tier_ranked_before_larger_medium_tier():
    alerts = prioritise([det("Bench", 0, 590, 0, 390), det("Stairs", 250, 300, 300, 320)])
    assert [a.class_name for a in alerts] == ["Stairs", "Bench"]


def test_within_tier_larger_box_first():
    alerts = prioritise([det("Car", 0, 50), det("Bus", 400, 600, 0, 400)])
    assert [a.class_name for a in alerts] == ["Bus", "Car"]


def test_context_classes_ignored():
    assert prioritise([det("Road"), det("Building")]) == []


def test_duplicates_in_same_region_collapse():
    alerts = prioritise([det("Person", 10, 50), det("Person", 60, 100, conf=0.5), det("Person", 500, 590)])
    assert sorted(a.region for a in alerts) == ["left", "right"]


def test_unknown_class_is_kept_not_dropped():
    assert prioritise([det("Scooter")])[0].class_name == "Scooter"


def test_message_wording():
    alerts = prioritise([det("Stairs", 250, 350), det("Teraffic Barrel", 0, 50)])
    assert compose_message(alerts) == "Caution. Stairs, centre. Traffic barrel, left."
    assert compose_message(prioritise([det("Pedestrian crosswalk", 250, 350)])) == "Crosswalk, centre."


def test_no_detections_means_silence_not_all_clear():
    assert compose_message([]) is None


def test_scheduler_limits_alerts_per_frame():
    s = AlertScheduler(max_alerts=2, cooldown_s=5)
    alerts = prioritise([det("Car", 0, 50), det("Stairs", 250, 350), det("Bench", 500, 590)])
    spoken, suppressed = s.select(alerts, now_s=0)
    assert len(spoken) == 2 and spoken[0].tier == HIGH
    assert [a.class_name for a in suppressed] == ["Bench"]


def test_scheduler_suppresses_repeats_until_cooldown_expires():
    s = AlertScheduler(max_alerts=2, cooldown_s=5)
    car = prioritise([det("Car")])
    assert s.select(car, now_s=0)[0]
    assert s.select(car, now_s=4.9)[0] == []
    assert s.select(car, now_s=5.0)[0] == car


def test_scheduler_same_class_new_region_is_not_suppressed():
    s = AlertScheduler(cooldown_s=5)
    s.select(prioritise([det("Car", 0, 50)]), now_s=0)
    spoken, _ = s.select(prioritise([det("Car", 500, 590)]), now_s=1)
    assert spoken[0].region == "right"


def test_suppressed_repeat_frees_slot_for_next_alert():
    s = AlertScheduler(max_alerts=1, cooldown_s=5)
    s.select(prioritise([det("Car")]), now_s=0)
    spoken, _ = s.select(prioritise([det("Car"), det("Bench", 500, 590)]), now_s=1)
    assert [a.class_name for a in spoken] == ["Bench"]


@pytest.mark.parametrize("kwargs", [{"max_alerts": 0}, {"cooldown_s": -1}])
def test_scheduler_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        AlertScheduler(**kwargs)

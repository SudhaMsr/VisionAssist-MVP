"""Rule-based hazard prioritisation and alert wording.

Deliberately simple and deterministic so every rule is unit-testable.

What this does NOT do:
- It does not estimate distance. Box size is used only to order alerts
  within the same priority tier.
- It does not give directions. "left/centre/right" is the horizontal third
  of the *image* the object's box centre falls in, not a heading for the user.
- It never says the path is clear. No alert means nothing alert-worthy was
  detected above the confidence threshold, which is not the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection

HIGH, MEDIUM, LOW = 1, 2, 3
TIER_LABELS = {HIGH: "high", MEDIUM: "medium", LOW: "low"}

# Every class in the ROD checkpoint must appear here or in IGNORED_CLASSES;
# tests enforce this so a new class cannot be silently dropped.
HAZARD_TIERS: dict[str, int] = {
    # Fall / trip hazards and moving vehicles.
    "Stairs": HIGH,
    "Manhole": HIGH,
    "Car": HIGH,
    "Motorcycle": HIGH,
    "Bike": HIGH,
    "Truck": HIGH,
    "Bus": HIGH,
    # Static or slow obstacles that may block a walking path.
    "Person": MEDIUM,
    "Dog": MEDIUM,
    "Traffic Cone": MEDIUM,
    "Teraffic Barrel": MEDIUM,
    "Electrical Pole": MEDIUM,
    "Dustbin": MEDIUM,
    "Bench": MEDIUM,
    "Chair": MEDIUM,
    "Plant Pot": MEDIUM,
    "Electrical Box": MEDIUM,
    "Fire hydrant": MEDIUM,
    "Guard rail": MEDIUM,
    "Bicycle Rack": MEDIUM,
    "Tree": MEDIUM,
    # Useful context, rarely an obstacle.
    "Pedestrian crosswalk": LOW,
    "Traffic sign": LOW,
}

# Scene context that would be present in nearly every frame.
IGNORED_CLASSES = frozenset({"Road", "Building"})

# Can walk into the user's path. Ranked before static objects of the same tier,
# whose boxes (a tree's canopy, for example) are often far larger.
MOVING_CLASSES = frozenset({"Person", "Dog"})

# Fall hazards. Ranked first in the high tier, so vehicle alerts (whose boxes are
# usually far larger) can never push a staircase or open manhole out of the alert slots.
FALL_HAZARDS = frozenset({"Stairs", "Manhole"})

# Spoken forms for labels that read badly (including the dataset's typo).
SPOKEN_NAMES = {
    "Teraffic Barrel": "traffic barrel",
    "Pedestrian crosswalk": "crosswalk",
    "Bike": "bicycle",
}


def spoken_name(class_name: str) -> str:
    return SPOKEN_NAMES.get(class_name, class_name.lower())


def image_region(det: Detection) -> str:
    """Horizontal third of the image containing the box centre."""
    x = det.center_x_frac
    if x < 1 / 3:
        return "left"
    if x > 2 / 3:
        return "right"
    return "centre"


@dataclass(frozen=True)
class Alert:
    class_name: str
    tier: int
    region: str
    confidence: float
    area_frac: float

    @property
    def key(self) -> tuple[str, str]:
        return (self.class_name, self.region)

    @property
    def phrase(self) -> str:
        return f"{spoken_name(self.class_name)}, {self.region}"

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "tier": TIER_LABELS[self.tier],
            "image_region": self.region,
            "confidence": round(self.confidence, 4),
            "box_area_frac": round(self.area_frac, 4),
        }


def prioritise(detections: list[Detection]) -> list[Alert]:
    """Collapse detections to one alert per (class, region), most urgent first.

    Order: tier, then fall hazards (stairs, manholes) first, then moving classes
    before static ones, then larger box, then higher confidence. Unknown classes are
    treated as MEDIUM rather than dropped, so a mislabelled checkpoint fails loud
    in speech rather than silently.
    """
    best: dict[tuple[str, str], Alert] = {}
    for det in detections:
        if det.class_name in IGNORED_CLASSES:
            continue
        alert = Alert(
            class_name=det.class_name,
            tier=HAZARD_TIERS.get(det.class_name, MEDIUM),
            region=image_region(det),
            confidence=det.confidence,
            area_frac=det.area_frac,
        )
        current = best.get(alert.key)
        if current is None or (alert.area_frac, alert.confidence) > (current.area_frac, current.confidence):
            best[alert.key] = alert
    return sorted(
        best.values(),
        key=lambda a: (
            a.tier, a.class_name not in FALL_HAZARDS, a.class_name not in MOVING_CLASSES, -a.area_frac, -a.confidence
        ),
    )


def compose_message(alerts: list[Alert]) -> str | None:
    if not alerts:
        return None
    body = ". ".join(a.phrase[0].upper() + a.phrase[1:] for a in alerts)
    prefix = "Caution. " if alerts[0].tier == HIGH else ""
    return f"{prefix}{body}."


@dataclass
class AlertScheduler:
    """Limits alerts per frame and suppresses repeats within a cooldown.

    Time is passed in explicitly (seconds) so behaviour is deterministic and
    testable; the demo uses its own timeline, a live system would use a clock.
    """

    max_alerts: int = 2
    cooldown_s: float = 6.0
    _last_spoken: dict[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self):
        if self.max_alerts < 1:
            raise ValueError("max_alerts must be >= 1")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")

    def select(self, alerts: list[Alert], now_s: float) -> tuple[list[Alert], list[Alert]]:
        """Return (to_speak, suppressed). `alerts` must already be prioritised.

        A lower-tier alert never takes the place of a higher-tier hazard that is
        only held back as a repeat: hearing "tree" alone while a motorcycle is
        still in view would misrepresent what matters most.
        """
        to_speak, suppressed = [], []
        repeat_held_tier = None
        for alert in alerts:
            last = self._last_spoken.get(alert.key)
            recently_spoken = last is not None and now_s - last < self.cooldown_s
            outranked = repeat_held_tier is not None and alert.tier > repeat_held_tier
            if recently_spoken or outranked or len(to_speak) >= self.max_alerts:
                suppressed.append(alert)
                if recently_spoken and repeat_held_tier is None:
                    repeat_held_tier = alert.tier
            else:
                to_speak.append(alert)
        for alert in to_speak:
            self._last_spoken[alert.key] = now_s
        return to_speak, suppressed

    def reset(self) -> None:
        self._last_spoken.clear()

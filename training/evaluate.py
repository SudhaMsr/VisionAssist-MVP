"""Compare detection systems on held-out sets and check the v1 acceptance criteria.

A system is "name=weights" or "name=weights+assist_weights" (the app's COCO assist), e.g.
    today=models/rod_yolov8n_50ep_best.pt
    today+assist=models/rod_yolov8n_50ep_best.pt+models/yolov8n.pt
    A=runs/A/weights/best.pt

Sets are image-list files written by build_dataset assemble (labels next to images):
test_rod (all 25 classes), test_phone (your held-out walks), test_coco (COCO-mapped classes only).

Per system and set:
  - recall / precision per class at the app's threshold (conf 0.35, IoU 0.5), through
    vision_assist.detector.Detector, so the assist is merged exactly as in the app;
  - Ultralytics mAP50 and mAP50-95 per class (single models only);
  - alert simulation (2 alerts per image, the app's rules) against the first system:
    fall hazards pushed out, high-tier alerts lost, people displaced by static objects;
  - median inference time.
Then the acceptance criteria from the plan are checked for each candidate.

Usage:
    python -m training.evaluate --work /kaggle/working/ft_v1 --system today=... --system A=... \
        --candidate A --device 0 --out /kaggle/working/eval_v1
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2

from training.common import CLASS_NAMES, PROTECTED_CLASSES, read_list, read_yolo_labels, to_xyxy, overlap

APP_CONF = 0.35
IOU = 0.5
COCO_EVAL_CLASSES = ["Person", "Car", "Bus", "Truck", "Motorcycle", "Dog", "Bench", "Chair", "Fire hydrant",
                     "Plant Pot"]
FOCUS = ["Car", "Person"]


def parse_system(spec: str) -> tuple[str, Path, Path | None]:
    name, _, weights = spec.partition("=")
    if not name or not weights:
        raise ValueError(f"Expected name=weights[+assist], got {spec!r}")
    main, _, assist = weights.partition("+")
    return name, Path(main), Path(assist) if assist else None


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")


def match(gt: list[tuple[int, tuple]], preds: list[tuple[int, float, tuple]], classes: set[str] | None) -> Counter:
    """Greedy same-class matching at IoU >= 0.5; returns tp/fp/fn counts keyed '<kind>:<class>'."""
    c = Counter()
    used = set()
    for cls, box in gt:
        if classes and CLASS_NAMES[cls] not in classes:
            continue
        best, j_best = 0.0, None
        for j, (pcls, _, pbox) in enumerate(preds):
            if pcls == cls and j not in used:
                iou = overlap(box, pbox)[0]
                if iou > best:
                    best, j_best = iou, j
        if best >= IOU:
            used.add(j_best)
            c[f"tp:{CLASS_NAMES[cls]}"] += 1
        else:
            c[f"fn:{CLASS_NAMES[cls]}"] += 1
    for j, (pcls, _, _) in enumerate(preds):
        if j not in used and (not classes or CLASS_NAMES[pcls] in classes):
            c[f"fp:{CLASS_NAMES[pcls]}"] += 1
    return c


def rates(counts: Counter, classes: list[str]) -> dict:
    out = {}
    for cls in classes:
        tp, fp, fn = counts[f"tp:{cls}"], counts[f"fp:{cls}"], counts[f"fn:{cls}"]
        out[cls] = {"labelled": tp + fn, "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                    "precision": round(tp / (tp + fp), 4) if tp + fp else None, "tp": tp, "fp": fp, "fn": fn}
    return out


def alert_keys(detections) -> list:
    from vision_assist.guidance import prioritise

    return prioritise(detections)[:2]


def run_system(name, weights, assist, images, classes, device, keep_alerts):
    from vision_assist.detector import Detector

    det = Detector(weights, conf=APP_CONF, device=device, assist_model_path=assist)
    counts, times, alerts = Counter(), [], {}
    for img in images:
        frame = cv2.imread(str(img))
        if frame is None:
            raise ValueError(f"Could not read {img}")
        h, w = frame.shape[:2]
        result = det.detect(frame)
        times.append(result.inference_ms)
        gt = [(r[0], to_xyxy(r, w, h)) for r in read_yolo_labels(label_path(img))]
        preds = [(CLASS_NAMES.index(d.class_name), d.confidence, d.bbox) for d in result.detections]
        counts += match(gt, preds, set(classes) if classes else None)
        if keep_alerts:
            alerts[img.name] = alert_keys(result.detections)
    return counts, statistics.median(times) if times else None, alerts


def ultralytics_map(weights: Path, data_yaml: Path, device) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="val", imgsz=640, batch=16, device=device,
                                plots=False, verbose=False)
    per = {CLASS_NAMES[c]: {"mAP50": round(float(m.box.ap50[i]), 4), "mAP50_95": round(float(m.box.ap[i]), 4)}
           for i, c in enumerate(m.box.ap_class_index)}
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4), "per_class": per}


def compare_alerts(ref: dict, other: dict) -> dict:
    from vision_assist.guidance import FALL_HAZARDS, HIGH, MOVING_CLASSES, MEDIUM

    c = Counter()
    for name, a in ref.items():
        b = other[name]
        ka, kb = {x.key for x in a}, {x.key for x in b}
        lost = ka - kb
        c["messages_changed"] += [x.key for x in a] != [x.key for x in b]
        c["fall_hazard_pushed_out"] += any(k[0] in FALL_HAZARDS for k in lost)
        c["high_tier_alert_lost"] += any(x.tier == HIGH for x in a if x.key in lost)
        c["person_or_dog_displaced_by_static"] += (
            any(x.class_name in MOVING_CLASSES for x in a if x.key in lost)
            and any(x.tier == MEDIUM and x.class_name not in MOVING_CLASSES for x in b if x.key not in ka))
    return dict(c)


def acceptance(report: dict, candidate: str, baseline: str, assist: str | None) -> dict:
    """The v1 pass/fail criteria (docs/phone_footage_spec.md and the training plan)."""
    checks = {}
    sets = report["sets"]
    if "test_phone" in sets:
        r = sets["test_phone"]["systems"]
        for cls in FOCUS:
            cand, base = r[candidate]["app_threshold"][cls], r[baseline]["app_threshold"][cls]
            if cand["recall"] is None or base["recall"] is None:
                checks[f"phone {cls} recall"] = "no labelled boxes"
                continue
            ok = cand["recall"] >= base["recall"] + 0.15
            if assist:
                ok = ok and cand["recall"] >= r[assist]["app_threshold"][cls]["recall"]
            prec_ok = (cand["precision"] or 0) >= (base["precision"] or 0) - 0.05
            checks[f"phone {cls} recall +0.15 vs {baseline}" + (f" and >= {assist}" if assist else "")] = (
                "PASS" if ok else "FAIL")
            checks[f"phone {cls} precision within 0.05 of {baseline}"] = "PASS" if prec_ok else "FAIL"
    else:
        checks["phone test set"] = "MISSING: no labelled phone test walks, real-world criteria cannot be checked"
    if "test_rod" in sets and "map" in sets["test_rod"]["systems"].get(candidate, {}):
        cand = sets["test_rod"]["systems"][candidate]["map"]["per_class"]
        base = sets["test_rod"]["systems"][baseline]["map"]["per_class"]
        drops = {c: round(cand[c]["mAP50"] - base[c]["mAP50"], 4) for c in base if c in cand}
        checks["ROD test: no class drops > 0.02 mAP50"] = (
            "PASS" if all(d >= -0.02 for d in drops.values()) else f"FAIL {[c for c, d in drops.items() if d < -0.02]}")
        protected_ok = all(drops.get(c, 0) >= -0.01 for c in PROTECTED_CLASSES)
        protected_detail = ", ".join(f"{c}: {drops.get(c)}" for c in PROTECTED_CLASSES)
        checks["ROD test: stairs, manhole, cone within 0.01"] = (
            "PASS" if protected_ok else f"FAIL {{{protected_detail}}}")
    alerts = report.get("alerts", {}).get(candidate)
    if alerts:
        checks["alerts: 0 fall hazards pushed out"] = "PASS" if alerts.get("fall_hazard_pushed_out", 0) == 0 else "FAIL"
        base_disp = report["alerts"].get("_baseline_person_displaced", 0)
        checks["alerts: people displaced by static <= baseline"] = (
            "PASS" if alerts.get("person_or_dog_displaced_by_static", 0) <= base_disp else "FAIL")
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True, help="Folder with test_*.txt and data_test_*.yaml from build_dataset")
    p.add_argument("--system", action="append", required=True, help="name=weights[+assist]; the first is the baseline")
    p.add_argument("--candidate", action="append", default=[], help="System name(s) to check against the criteria")
    p.add_argument("--assist-baseline", default=None, help="Name of the today+assist system, if given")
    p.add_argument("--sets", default="test_rod,test_phone,test_coco")
    p.add_argument("--limit", type=int, default=None, help="Images per set (smoke tests only)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    systems = [parse_system(s) for s in args.system]
    baseline = systems[0][0]
    systems_repr = {n: [str(w), str(a) if a else None] for n, w, a in systems}
    report = {"app_threshold": APP_CONF, "iou": IOU, "systems": systems_repr, "sets": {}, "alerts": {}}
    for set_name in args.sets.split(","):
        list_file = work / f"{set_name}.txt"
        images = [Path(x) for x in read_list(list_file)] if list_file.exists() else []
        if not images:
            continue
        images = images[: args.limit] if args.limit else images
        classes = COCO_EVAL_CLASSES if set_name == "test_coco" else None
        report_classes = classes or CLASS_NAMES
        entry = {"images": len(images), "systems": {}}
        all_alerts = {}
        for name, weights, assist in systems:
            counts, median_ms, alerts = run_system(name, weights, assist, images, classes, args.device,
                                                   keep_alerts=set_name == "test_rod")
            sysrep = {"app_threshold": rates(counts, report_classes), "median_inference_ms": median_ms}
            data_yaml = work / f"data_{set_name}.yaml"
            if assist is None and data_yaml.exists() and not args.limit:
                sysrep["map"] = ultralytics_map(weights, data_yaml, args.device)
            entry["systems"][name] = sysrep
            all_alerts[name] = alerts
        report["sets"][set_name] = entry
        if set_name == "test_rod":
            for name, _, _ in systems[1:]:
                report["alerts"][name] = compare_alerts(all_alerts[baseline], all_alerts[name])
            report["alerts"]["_baseline_person_displaced"] = 0  # by definition, relative to itself
    report["acceptance"] = {c: acceptance(report, c, baseline, args.assist_baseline) for c in args.candidate}
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "eval_report.md").write_text(markdown(report))
    print((out / "eval_report.md").read_text())
    return 0


def markdown(report: dict) -> str:
    lines = [f"# Evaluation (conf {report['app_threshold']}, IoU {report['iou']})", ""]
    for set_name, entry in report["sets"].items():
        names = list(entry["systems"])
        focus = ["Car", "Person", *PROTECTED_CLASSES] if set_name != "test_coco" else ["Car", "Person", "Bus", "Truck"]
        lines += [f"## {set_name} ({entry['images']} images)", "",
                  "| class | " + " | ".join(f"{n} recall / precision" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for cls in focus:
            cells = []
            for n in names:
                r = entry["systems"][n]["app_threshold"].get(cls, {})
                cells.append(f"{r.get('recall')} / {r.get('precision')} (n={r.get('labelled')})")
            lines.append(f"| {cls} | " + " | ".join(cells) + " |")
        lines += ["", "| system | mAP50 | mAP50-95 | median ms |", "|---|---|---|---|"]
        for n in names:
            s = entry["systems"][n]
            m = s.get("map", {})
            ms = s['median_inference_ms'] or 0
            lines.append(f"| {n} | {m.get('mAP50', 'n/a')} | {m.get('mAP50_95', 'n/a')} | {ms:.1f} |")
        lines.append("")
    if report["alerts"]:
        cols = list(next(iter(v for k, v in report["alerts"].items() if not k.startswith("_")), {}))
        lines += ["## Alert simulation vs baseline (ROD test)", "", "| system | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for n, v in report["alerts"].items():
            if not n.startswith("_"):
                lines.append(f"| {n} | " + " | ".join(str(x) for x in v.values()) + " |")
        lines.append("")
    for cand, checks in report["acceptance"].items():
        lines += [f"## Acceptance: {cand}", ""] + [f"- {k}: **{v}**" for k, v in checks.items()] + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Independently verify the complete 20-country FTW release."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


COUNTRIES = (
    "austria",
    "belgium",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
)

COUNT_KEYS = (
    "pixel_tp",
    "pixel_fp",
    "pixel_fn",
    "object_tp",
    "object_fp",
    "object_fn",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_metrics(reports: dict[str, dict], result_key: str) -> dict[str, float | int]:
    totals = {
        key: sum(int(reports[country][result_key][key]) for country in COUNTRIES)
        for key in COUNT_KEYS
    }
    pixel_precision = totals["pixel_tp"] / (totals["pixel_tp"] + totals["pixel_fp"])
    pixel_recall = totals["pixel_tp"] / (totals["pixel_tp"] + totals["pixel_fn"])
    pixel_iou = totals["pixel_tp"] / (
        totals["pixel_tp"] + totals["pixel_fp"] + totals["pixel_fn"]
    )
    object_precision = totals["object_tp"] / (
        totals["object_tp"] + totals["object_fp"]
    )
    object_recall = totals["object_tp"] / (
        totals["object_tp"] + totals["object_fn"]
    )
    object_f1 = 2 * object_precision * object_recall / (
        object_precision + object_recall
    )
    return {
        "pixel_iou": pixel_iou,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
        **totals,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "results" / "ftw_full_20_country_metrics.csv"
    report_path = root / "results" / "full_submission_report.json"

    reports = {}
    for country in COUNTRIES:
        validation_path = root / "reports" / "validation" / f"{country}.json"
        test_path = root / "reports" / "test" / f"{country}.json"
        if not validation_path.exists() or not test_path.exists():
            raise RuntimeError(f"Missing validation or test evidence for {country}")
        report = json.loads(test_path.read_text(encoding="utf-8"))
        if report.get("country") != country:
            raise RuntimeError(f"Country mismatch in {test_path}")
        reports[country] = report

    baseline = pooled_metrics(reports, "baseline_test")
    optimized = pooled_metrics(reports, "frozen_test")
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    for name, calculated in (("baseline_aggregate", baseline), ("optimized_aggregate", optimized)):
        recorded = summary[name]
        for key, value in calculated.items():
            if isinstance(value, int):
                if int(recorded[key]) != value:
                    raise RuntimeError(f"Count mismatch for {name}.{key}")
            elif not math.isclose(float(recorded[key]), value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Metric mismatch for {name}.{key}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = list(COUNTRIES) + ["All countries"]
    if [row["Country"] for row in rows] != expected_rows:
        raise RuntimeError("CSV country rows are incomplete or out of order")
    all_row = rows[-1]
    csv_metric_map = {
        "PixelIoU": "pixel_iou",
        "PixelPrecision": "pixel_precision",
        "PixelRecall": "pixel_recall",
        "ObjectPrecision": "object_precision",
        "ObjectRecall": "object_recall",
        "ObjectF1": "object_f1",
    }
    for column, key in csv_metric_map.items():
        if not math.isclose(
            float(all_row[column]), float(optimized[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"CSV All countries mismatch for {column}")

    expected_f1 = 0.5321830716312617
    if not math.isclose(optimized["object_f1"], expected_f1, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Unexpected optimized Object F1")

    print("VERIFIED")
    print(f"countries={len(COUNTRIES)}")
    print(f"baseline_object_f1={baseline['object_f1']:.15f}")
    print(f"optimized_object_f1={optimized['object_f1']:.15f}")
    print(f"absolute_gain={optimized['object_f1'] - baseline['object_f1']:.15f}")
    print(f"csv_sha256={sha256(csv_path)}")
    print(f"report_sha256={sha256(report_path)}")


if __name__ == "__main__":
    main()

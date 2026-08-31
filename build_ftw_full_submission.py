#!/usr/bin/env python
"""Build the honest official-scope FTW CSV from frozen country test reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FULL_DATA_COUNTRIES = (
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

CSV_FIELDS = (
    "Country",
    "PixelIoU",
    "PixelPrecision",
    "PixelRecall",
    "ObjectPrecision",
    "ObjectRecall",
    "ObjectF1",
)

METRIC_MAP = {
    "PixelIoU": "pixel_iou",
    "PixelPrecision": "pixel_precision",
    "PixelRecall": "pixel_recall",
    "ObjectPrecision": "object_precision",
    "ObjectRecall": "object_recall",
    "ObjectF1": "object_f1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = {}
    missing = []
    for country in FULL_DATA_COUNTRIES:
        path = args.results_root / country / "frozen" / "frozen_test_report.json"
        if not path.exists():
            missing.append(country)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("country") != country:
            raise RuntimeError(f"Country mismatch in {path}")
        reports[country] = report
    if missing:
        raise RuntimeError(
            "Refusing to build a partial leaderboard submission; missing: "
            + ", ".join(missing)
        )

    rows = []
    for country in FULL_DATA_COUNTRIES:
        result = reports[country]["frozen_test"]
        rows.append(
            {
                "Country": country,
                **{csv_name: result[key] for csv_name, key in METRIC_MAP.items()},
            }
        )
    def aggregate(result_key: str) -> dict:
        totals = {
            key: sum(
                int(reports[c][result_key][key]) for c in FULL_DATA_COUNTRIES
            )
            for key in (
                "pixel_tp", "pixel_fp", "pixel_fn",
                "object_tp", "object_fp", "object_fn",
            )
        }
        pixel_precision = totals["pixel_tp"] / max(
            totals["pixel_tp"] + totals["pixel_fp"], 1
        )
        pixel_recall = totals["pixel_tp"] / max(
            totals["pixel_tp"] + totals["pixel_fn"], 1
        )
        pixel_iou = totals["pixel_tp"] / max(
            totals["pixel_tp"] + totals["pixel_fp"] + totals["pixel_fn"], 1
        )
        object_precision = totals["object_tp"] / max(
            totals["object_tp"] + totals["object_fp"], 1
        )
        object_recall = totals["object_tp"] / max(
            totals["object_tp"] + totals["object_fn"], 1
        )
        object_f1 = 2 * object_precision * object_recall / max(
            object_precision + object_recall, 1e-12
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

    baseline_aggregate = aggregate("baseline_test")
    optimized_aggregate = aggregate("frozen_test")
    all_row = {
        "Country": "All countries",
        **{
            csv_name: optimized_aggregate[key]
            for csv_name, key in METRIC_MAP.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ftw_full_20_country_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(all_row)
    summary = {
        "benchmark": "Fields of the World",
        "scope": "20 FTW full-data countries",
        "aggregation": "official pooled TP/FP/FN across all full-data test samples",
        "countries": list(FULL_DATA_COUNTRIES),
        "baseline_aggregate": baseline_aggregate,
        "optimized_aggregate": optimized_aggregate,
        "object_f1_absolute_gain": (
            optimized_aggregate["object_f1"] - baseline_aggregate["object_f1"]
        ),
        "leaderboard_csv": str(csv_path),
    }
    (args.output_dir / "full_submission_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate one validation-frozen FTW country configuration on test once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from optimize_ftw import Candidate, decision_mask, metrics
from optimize_ftw_v2 import RefinedCandidate, refine_components
from optimize_ftw_v3 import (
    InstanceCandidate,
    grow_instances,
    split_large_components,
)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def predict(cache: dict[str, np.ndarray], base: Candidate, instance: InstanceCandidate):
    probabilities = cache[base.ensemble].astype(np.float32, copy=False)
    raw = decision_mask(probabilities, base)
    refined = RefinedCandidate(
        secondary="none",
        secondary_weight=0.0,
        field_threshold=base.field_threshold,
        background_ratio=base.background_ratio,
        boundary_ratio=base.boundary_ratio,
        min_object_pixels=base.min_object_pixels,
        rescue_mean_field=1.1,
    )
    base_prediction = refine_components(raw, probabilities[:, 1], refined)
    if (
        instance.large_min_pixels
        or instance.large_boundary_ratio
        or instance.large_open_radius
    ):
        split = split_large_components(
            base_prediction, probabilities, refined, instance
        )
    else:
        split = base_prediction
    return grow_instances(split, probabilities, instance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    country = selection["country"]
    base = Candidate(**selection["selected_base_configuration"])
    instance = InstanceCandidate(**selection["selected_instance_configuration"])
    cache = load_cache(args.test_cache)
    if base.ensemble not in cache:
        raise RuntimeError(
            f"Frozen ensemble {base.ensemble!r} is missing from test cache"
        )
    baseline = Candidate("base", 0.0, 1.0, 1.0)
    baseline_result = metrics(
        cache["masks"],
        decision_mask(cache["base"], baseline),
        exact_objects=True,
    )
    prediction = predict(cache, base, instance)
    result = metrics(cache["masks"], prediction, exact_objects=True)
    report = {
        "benchmark": "Fields of the World",
        "country": country,
        "selection_protocol": "configuration frozen on validation before test cache evaluation",
        "selected_base_configuration": selection["selected_base_configuration"],
        "selected_instance_configuration": selection["selected_instance_configuration"],
        "validation": selection["selected_validation"],
        "baseline_test": baseline_result,
        "frozen_test": result,
        "object_f1_absolute_gain": result["object_f1"] - baseline_result["object_f1"],
        "pixel_iou_absolute_gain": result["pixel_iou"] - baseline_result["pixel_iou"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "leaderboard_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Country",
                "PixelIoU",
                "PixelPrecision",
                "PixelRecall",
                "ObjectPrecision",
                "ObjectRecall",
                "ObjectF1",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "Country": country,
                "PixelIoU": result["pixel_iou"],
                "PixelPrecision": result["pixel_precision"],
                "PixelRecall": result["pixel_recall"],
                "ObjectPrecision": result["object_precision"],
                "ObjectRecall": result["object_recall"],
                "ObjectF1": result["object_f1"],
            }
        )
    np.savez_compressed(
        args.output_dir / "predictions_test.npz",
        predictions=prediction.astype(np.uint8),
        masks=cache["masks"].astype(np.uint8),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

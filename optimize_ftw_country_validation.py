#!/usr/bin/env python
"""Validation-only country optimizer for the FTW multi-country submission."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from optimize_ftw import Candidate, decision_mask, metrics
from optimize_ftw_v2 import RefinedCandidate, refine_components
from optimize_ftw_v3 import InstanceCandidate, grow_instances, split_large_components


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def rank(item: dict) -> tuple[float, float]:
    return item["object_f1"], item["pixel_iou"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iou-drop", type=float, default=0.05)
    parser.add_argument("--search-max-samples", type=int, default=64)
    args = parser.parse_args()

    full_cache = load_cache(args.validation_cache)
    full_masks = full_cache["masks"]
    if len(full_masks) > args.search_max_samples:
        search_indices = np.linspace(
            0, len(full_masks) - 1, args.search_max_samples, dtype=int
        )
        cache = {
            name: values[search_indices]
            for name, values in full_cache.items()
        }
    else:
        search_indices = np.arange(len(full_masks))
        cache = full_cache
    ensembles = [name for name in cache if name != "masks"]
    masks = cache["masks"]
    baseline = Candidate("base", 0.0, 1.0, 1.0)
    baseline_prediction = decision_mask(full_cache["base"], baseline)
    baseline_result = metrics(full_masks, baseline_prediction, exact_objects=True)
    search_baseline = metrics(
        masks,
        decision_mask(cache["base"], baseline),
        exact_objects=False,
    )
    minimum_iou = max(0.0, search_baseline["pixel_iou"] - args.max_iou_drop)
    full_minimum_iou = max(
        0.0, baseline_result["pixel_iou"] - args.max_iou_drop
    )

    base_records: list[dict] = []

    def evaluate_base(candidate: Candidate) -> dict:
        # Candidate masks can be several MiB even on the search subset.  They
        # are scored once, so retaining hundreds of them only creates memory
        # pressure without saving work.
        prediction = decision_mask(cache[candidate.ensemble], candidate)
        result = metrics(masks, prediction, exact_objects=False)
        record = {**asdict(candidate), **result}
        base_records.append(record)
        return record

    # Coarse semantic decision search.
    for ensemble in ensembles:
        for field_threshold in (0.0, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
            for background_ratio in (0.60, 0.80, 1.00, 1.20):
                for boundary_ratio in (0.80, 1.00, 1.20, 1.50, 2.00, 3.00):
                    evaluate_base(
                        Candidate(
                            ensemble,
                            field_threshold,
                            background_ratio,
                            boundary_ratio,
                        )
                    )
    eligible_base = [r for r in base_records if r["pixel_iou"] >= minimum_iou]
    eligible_base.sort(key=rank, reverse=True)

    # Filter small false-positive components around the strongest raw masks.
    seed_signatures = []
    for record in eligible_base:
        signature = (
            record["ensemble"],
            record["field_threshold"],
            record["background_ratio"],
            record["boundary_ratio"],
        )
        if signature not in seed_signatures:
            seed_signatures.append(signature)
        if len(seed_signatures) == 6:
            break
    for ensemble, field_threshold, background_ratio, boundary_ratio in seed_signatures:
        for minimum in (4, 8, 16, 24, 32, 48, 64, 96, 128):
            evaluate_base(
                Candidate(
                    ensemble,
                    field_threshold,
                    background_ratio,
                    boundary_ratio,
                    min_object_pixels=minimum,
                )
            )
    eligible_base = [r for r in base_records if r["pixel_iou"] >= minimum_iou]
    eligible_base.sort(key=rank, reverse=True)
    full_base_records = []
    seen_base = set()
    for item in eligible_base:
        candidate = Candidate(
            **{name: item[name] for name in asdict(baseline)}
        )
        signature = tuple(asdict(candidate).values())
        if signature in seen_base:
            continue
        seen_base.add(signature)
        result = metrics(
            full_masks,
            decision_mask(full_cache[candidate.ensemble], candidate),
            exact_objects=False,
        )
        full_base_records.append({**asdict(candidate), **result})
        if len(full_base_records) == 20:
            break
    eligible_full_base = [
        item for item in full_base_records
        if item["pixel_iou"] >= full_minimum_iou
    ]
    eligible_full_base.sort(key=rank, reverse=True)
    best_base_record = eligible_full_base[0]
    best_base = Candidate(
        **{name: best_base_record[name] for name in asdict(baseline)}
    )

    probabilities = cache[best_base.ensemble].astype(np.float32, copy=False)
    field = probabilities[:, 1]
    raw = decision_mask(probabilities, best_base)
    base_refined = RefinedCandidate(
        secondary="none",
        secondary_weight=0.0,
        field_threshold=best_base.field_threshold,
        background_ratio=best_base.background_ratio,
        boundary_ratio=best_base.boundary_ratio,
        min_object_pixels=best_base.min_object_pixels,
        rescue_mean_field=1.1,
    )
    # Recreate the selected base with the component filter used by V3.
    base_prediction = refine_components(raw, field, base_refined)

    instance_records: dict[tuple, dict] = {}
    split_cache: dict[tuple[int, float, int], np.ndarray] = {}

    def instance_key(candidate: InstanceCandidate) -> tuple:
        return tuple(asdict(candidate).values())

    def evaluate_instance(
        candidate: InstanceCandidate, retain_split: bool = True
    ) -> dict:
        key = instance_key(candidate)
        if key in instance_records:
            return instance_records[key]
        split_key = (
            candidate.large_min_pixels,
            candidate.large_boundary_ratio,
            candidate.large_open_radius,
        )
        if split_key in split_cache:
            split_prediction = split_cache[split_key]
        else:
            if split_key == (0, 0.0, 0):
                split_prediction = base_prediction
            else:
                split_prediction = split_large_components(
                    base_prediction,
                    probabilities,
                    base_refined,
                    InstanceCandidate(*split_key),
                )
            if retain_split:
                split_cache[split_key] = split_prediction
        prediction = grow_instances(split_prediction, probabilities, candidate)
        record = {**asdict(candidate), **metrics(masks, prediction, exact_objects=False)}
        instance_records[key] = record
        return record

    evaluate_instance(InstanceCandidate(), retain_split=False)
    for minimum in (128, 192, 256, 384, 512, 768, 1024):
        for boundary_ratio in (2.0, 3.0, 4.0, 6.0, 10.0, 15.0):
            for open_radius in (0, 1, 2, 3):
                evaluate_instance(
                    InstanceCandidate(minimum, boundary_ratio, open_radius),
                    retain_split=False,
                )
    split_records = [
        r for r in instance_records.values() if r["grow_radius"] == 0
    ]
    split_records = [r for r in split_records if r["pixel_iou"] >= minimum_iou]
    split_records.sort(key=rank, reverse=True)

    # Two-pass coordinate search around each of four split seeds.
    axes = (
        ("grow_radius", (0, 2, 4, 6, 8, 10)),
        ("grow_field_threshold", (0.03, 0.05, 0.10, 0.15, 0.20, 0.30)),
        ("grow_background_ratio", (0.0, 0.20, 0.40, 0.60, 0.80)),
        ("grow_boundary_ratio", (0.0, 0.20, 0.50, 0.80, 1.00, 1.30)),
    )
    for seed in split_records[:4]:
        split_values = (
            int(seed["large_min_pixels"]),
            float(seed["large_boundary_ratio"]),
            int(seed["large_open_radius"]),
        )
        growth = {
            "grow_radius": 6,
            "grow_field_threshold": 0.10,
            "grow_background_ratio": 0.40,
            "grow_boundary_ratio": 0.0,
        }
        for _ in range(2):
            for field_name, choices in axes:
                candidates = []
                for choice in choices:
                    trial = dict(growth)
                    trial[field_name] = choice
                    record = evaluate_instance(
                        InstanceCandidate(*split_values, **trial)
                    )
                    if record["pixel_iou"] >= minimum_iou:
                        candidates.append(record)
                if candidates:
                    chosen = max(candidates, key=rank)
                    growth = {name: chosen[name] for name, _ in axes}

    eligible_instance = [
        r for r in instance_records.values() if r["pixel_iou"] >= minimum_iou
    ]
    eligible_instance.sort(key=rank, reverse=True)
    # Re-rank the leading instance candidates on every validation image.
    full_probabilities = full_cache[best_base.ensemble].astype(
        np.float32, copy=False
    )
    full_raw = decision_mask(full_probabilities, best_base)
    full_base_prediction = refine_components(
        full_raw, full_probabilities[:, 1], base_refined
    )
    full_split_cache: dict[tuple[int, float, int], np.ndarray] = {}
    full_instance_records = []
    for item in eligible_instance[:24]:
        candidate = InstanceCandidate(
            **{
                name: item[name]
                for name in asdict(InstanceCandidate())
            }
        )
        split_key = (
            candidate.large_min_pixels,
            candidate.large_boundary_ratio,
            candidate.large_open_radius,
        )
        if split_key not in full_split_cache:
            if split_key == (0, 0.0, 0):
                full_split_cache[split_key] = full_base_prediction
            else:
                full_split_cache[split_key] = split_large_components(
                    full_base_prediction,
                    full_probabilities,
                    base_refined,
                    InstanceCandidate(*split_key),
                )
        prediction = grow_instances(
            full_split_cache[split_key], full_probabilities, candidate
        )
        result = metrics(full_masks, prediction, exact_objects=False)
        full_instance_records.append({**asdict(candidate), **result})
    eligible_full_instance = [
        item for item in full_instance_records
        if item["pixel_iou"] >= full_minimum_iou
    ]
    eligible_full_instance.sort(key=rank, reverse=True)
    selected_instance_record = eligible_full_instance[0]
    selected_instance = InstanceCandidate(
        **{
            name: selected_instance_record[name]
            for name in asdict(InstanceCandidate())
        }
    )
    selected_split_key = (
        selected_instance.large_min_pixels,
        selected_instance.large_boundary_ratio,
        selected_instance.large_open_radius,
    )
    selected_prediction = grow_instances(
        full_split_cache[selected_split_key], full_probabilities, selected_instance
    )
    selected_exact = metrics(
        full_masks, selected_prediction, exact_objects=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "base_validation_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_records[0]))
        writer.writeheader()
        writer.writerows(sorted(base_records, key=rank, reverse=True))
    ordered_instances = sorted(instance_records.values(), key=rank, reverse=True)
    with (args.output_dir / "instance_validation_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered_instances[0]))
        writer.writeheader()
        writer.writerows(ordered_instances)
    report = {
        "benchmark": "Fields of the World",
        "country": args.country,
        "selection_protocol": "validation-only country-specific spatial-TTA pipeline",
        "minimum_pixel_iou": full_minimum_iou,
        "validation_samples": len(full_masks),
        "search_samples": len(search_indices),
        "baseline_configuration": asdict(baseline),
        "baseline_validation": baseline_result,
        "selected_base_configuration": asdict(best_base),
        "selected_instance_configuration": asdict(selected_instance),
        "selected_validation": selected_exact,
        "base_candidate_count": len(base_records),
        "instance_candidate_count": len(instance_records),
        "nearby_top_ten": eligible_instance[:10],
    }
    (args.output_dir / "country_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "nearby_top_ten"}, indent=2))


if __name__ == "__main__":
    main()

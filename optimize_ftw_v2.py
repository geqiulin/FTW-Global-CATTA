#!/usr/bin/env python
"""Second-stage validation-only refinement for the FTW Vietnam pipeline.

This script reuses cached official-model probabilities.  It searches weighted
inference blends and confidence-aware component filtering on validation only.
The test split is evaluated only when ``--evaluate-test`` is explicitly set
after the selected configuration has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, opening

from optimize_ftw import metrics


@dataclass(frozen=True)
class RefinedCandidate:
    secondary: str
    secondary_weight: float
    field_threshold: float
    background_ratio: float
    boundary_ratio: float
    min_object_pixels: int = 0
    rescue_mean_field: float = 1.1
    component_mean_field: float = 0.0
    open_small_max_pixels: int = 0


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as cached:
        return {key: cached[key] for key in cached.files}


def blended_probabilities(cache: dict[str, np.ndarray], candidate: RefinedCandidate):
    primary = cache["full8"].astype(np.float32, copy=False)
    if candidate.secondary == "none" or candidate.secondary_weight == 0:
        return primary
    secondary = cache[candidate.secondary].astype(np.float32, copy=False)
    weight = np.float32(candidate.secondary_weight)
    return primary * (1.0 - weight) + secondary * weight


def raw_mask(probabilities: np.ndarray, candidate: RefinedCandidate) -> np.ndarray:
    background = probabilities[:, 0]
    field = probabilities[:, 1]
    boundary = probabilities[:, 2]
    return (
        (field >= candidate.field_threshold)
        & (field >= candidate.background_ratio * background)
        & (field >= candidate.boundary_ratio * boundary)
    )


def refine_components(
    raw: np.ndarray, field_probability: np.ndarray, candidate: RefinedCandidate
) -> np.ndarray:
    needs_filter = (
        candidate.min_object_pixels > 0
        or candidate.component_mean_field > 0
        or candidate.open_small_max_pixels != 0
    )
    if not needs_filter:
        return raw

    result = np.zeros_like(raw)
    structure = ndimage.generate_binary_structure(2, 1)
    kernel = disk(1)
    for index, sample in enumerate(raw):
        labels, count = ndimage.label(sample, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        field_sums = np.bincount(
            labels.ravel(),
            weights=field_probability[index].ravel(),
            minlength=count + 1,
        )
        means = np.divide(
            field_sums,
            sizes,
            out=np.zeros_like(field_sums, dtype=np.float64),
            where=sizes > 0,
        )
        keep = np.ones(count + 1, dtype=bool)
        keep[0] = False
        if candidate.min_object_pixels:
            keep &= (sizes >= candidate.min_object_pixels) | (
                means >= candidate.rescue_mean_field
            )
        if candidate.component_mean_field:
            keep &= means >= candidate.component_mean_field
        kept = keep[labels]

        if candidate.open_small_max_pixels:
            opened = opening(kept, footprint=kernel)
            if candidate.open_small_max_pixels < 0:
                kept = opened
            else:
                preserve = sizes > candidate.open_small_max_pixels
                preserve[0] = False
                kept = opened | (kept & preserve[labels])
        result[index] = kept
    return result


def predict(cache: dict[str, np.ndarray], candidate: RefinedCandidate) -> np.ndarray:
    probabilities = blended_probabilities(cache, candidate)
    raw = raw_mask(probabilities, candidate)
    return refine_components(raw, probabilities[:, 1], candidate)


def score(cache: dict[str, np.ndarray], candidate: RefinedCandidate):
    prediction = predict(cache, candidate)
    return metrics(cache["masks"], prediction, exact_objects=False)


def record(candidate: RefinedCandidate, result: dict) -> dict:
    return {**asdict(candidate), **result}


def candidate_from_record(item: dict) -> RefinedCandidate:
    keys = asdict(RefinedCandidate("none", 0, 0, 0, 0))
    return RefinedCandidate(**{key: item[key] for key in keys})


def rank_key(item: dict, minimum_iou: float):
    eligible = item["pixel_iou"] >= minimum_iou
    return (eligible, item["object_f1"], item["pixel_iou"])


def unique_candidates(candidates):
    seen = set()
    for candidate in candidates:
        values = tuple(asdict(candidate).values())
        if values not in seen:
            seen.add(values)
            yield candidate


def search(validation: dict[str, np.ndarray], minimum_iou: float):
    blend_specs = [("none", 0.0)]
    blend_specs += [
        (name, weight)
        for name in ("scale2_spatial4", "spatial4", "temporal2", "base")
        for weight in (0.1, 0.2, 0.3, 0.4)
    ]

    coarse = []
    for secondary, weight in blend_specs:
        for field_threshold in (0.34, 0.38, 0.42, 0.46, 0.50):
            for background_ratio in (0.70, 0.80, 0.90, 1.00):
                for boundary_ratio in (1.30, 1.40, 1.50, 1.60, 1.70, 1.80):
                    candidate = RefinedCandidate(
                        secondary,
                        weight,
                        field_threshold,
                        background_ratio,
                        boundary_ratio,
                    )
                    coarse.append(record(candidate, score(validation, candidate)))
    coarse.sort(key=lambda item: rank_key(item, minimum_iou), reverse=True)

    fine_seeds = []
    for item in coarse:
        candidate = candidate_from_record(item)
        blend = (candidate.secondary, candidate.secondary_weight)
        if blend not in [
            (seed.secondary, seed.secondary_weight) for seed in fine_seeds
        ]:
            fine_seeds.append(candidate)
        if len(fine_seeds) == 8:
            break

    fine_candidates = []
    for seed in fine_seeds:
        for field_threshold in np.arange(
            seed.field_threshold - 0.04, seed.field_threshold + 0.041, 0.02
        ):
            for background_ratio in np.arange(
                seed.background_ratio - 0.10, seed.background_ratio + 0.101, 0.05
            ):
                for boundary_ratio in np.arange(
                    seed.boundary_ratio - 0.20, seed.boundary_ratio + 0.201, 0.05
                ):
                    fine_candidates.append(
                        RefinedCandidate(
                            seed.secondary,
                            seed.secondary_weight,
                            round(float(field_threshold), 3),
                            round(float(background_ratio), 3),
                            round(float(boundary_ratio), 3),
                        )
                    )
    fine = [record(candidate, score(validation, candidate)) for candidate in unique_candidates(fine_candidates)]
    all_raw = coarse + fine
    all_raw.sort(key=lambda item: rank_key(item, minimum_iou), reverse=True)

    filter_seeds = []
    for item in all_raw:
        candidate = candidate_from_record(item)
        signature = (
            candidate.secondary,
            candidate.secondary_weight,
            candidate.field_threshold,
            candidate.background_ratio,
            candidate.boundary_ratio,
        )
        if signature not in [
            (
                seed.secondary,
                seed.secondary_weight,
                seed.field_threshold,
                seed.background_ratio,
                seed.boundary_ratio,
            )
            for seed in filter_seeds
        ]:
            filter_seeds.append(candidate)
        if len(filter_seeds) == 8:
            break

    refined_candidates = []
    for seed in filter_seeds:
        for min_pixels in (0, 8, 16, 24, 32, 48):
            rescue_values = (1.1,) if min_pixels == 0 else (0.60, 0.75, 0.90, 1.1)
            for rescue in rescue_values:
                for component_mean in (0.0, 0.50, 0.60):
                    for open_max in (0, 128, -1):
                        refined_candidates.append(
                            RefinedCandidate(
                                seed.secondary,
                                seed.secondary_weight,
                                seed.field_threshold,
                                seed.background_ratio,
                                seed.boundary_ratio,
                                min_pixels,
                                rescue,
                                component_mean,
                                open_max,
                            )
                        )
    refined = [
        record(candidate, score(validation, candidate))
        for candidate in unique_candidates(refined_candidates)
    ]
    all_records = all_raw + refined
    all_records.sort(key=lambda item: rank_key(item, minimum_iou), reverse=True)
    eligible = [item for item in all_records if item["pixel_iou"] >= minimum_iou]
    return candidate_from_record(eligible[0]), all_records


def write_csv(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.6547743739171523)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    validation = load_cache(args.validation_cache)
    selected, records = search(validation, args.minimum_iou)
    validation_result = metrics(
        validation["masks"], predict(validation, selected), exact_objects=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "v2_validation_search.csv", records)
    report = {
        "selection_protocol": "validation-only selection; test requires explicit flag",
        "minimum_validation_pixel_iou": args.minimum_iou,
        "selected_configuration": asdict(selected),
        "selected_validation": validation_result,
    }
    if args.evaluate_test:
        if args.test_cache is None:
            raise ValueError("--test-cache is required with --evaluate-test")
        test = load_cache(args.test_cache)
        report["frozen_test"] = metrics(
            test["masks"], predict(test, selected), exact_objects=True
        )
    with (args.output_dir / "v2_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

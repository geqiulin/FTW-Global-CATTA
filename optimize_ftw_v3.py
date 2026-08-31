#!/usr/bin/env python
"""Validation-only instance-boundary refinement for FTW Vietnam V3."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import dilation, disk, opening
from skimage.segmentation import watershed

from optimize_ftw import metrics
from optimize_ftw_v2 import (
    RefinedCandidate,
    blended_probabilities,
    load_cache,
    predict,
    refine_components,
)


@dataclass(frozen=True)
class InstanceCandidate:
    large_min_pixels: int = 0
    large_boundary_ratio: float = 0.0
    large_open_radius: int = 0
    grow_radius: int = 0
    grow_field_threshold: float = 0.0
    grow_background_ratio: float = 0.0
    grow_boundary_ratio: float = 0.0


def load_v2_candidate(path: Path) -> RefinedCandidate:
    report = json.loads(path.read_text(encoding="utf-8"))
    names = {field.name for field in fields(RefinedCandidate)}
    return RefinedCandidate(
        **{
            key: value
            for key, value in report["selected_configuration"].items()
            if key in names
        }
    )


def split_large_components(
    prediction: np.ndarray,
    probabilities: np.ndarray,
    base_candidate: RefinedCandidate,
    candidate: InstanceCandidate,
) -> np.ndarray:
    if not (
        candidate.large_min_pixels
        and (candidate.large_boundary_ratio or candidate.large_open_radius)
    ):
        return prediction

    result = np.empty_like(prediction)
    structure = ndimage.generate_binary_structure(2, 1)
    field = probabilities[:, 1]
    boundary = probabilities[:, 2]
    kernel = disk(candidate.large_open_radius) if candidate.large_open_radius else None
    for index, sample in enumerate(prediction):
        labels, count = ndimage.label(sample, structure=structure)
        if count == 0:
            result[index] = sample
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        large = (labels > 0) & (sizes[labels] >= candidate.large_min_pixels)
        refined = sample.copy()
        if candidate.large_boundary_ratio:
            stronger_boundary = (
                field[index] >= candidate.large_boundary_ratio * boundary[index]
            )
            refined[large & ~stronger_boundary] = False
        if kernel is not None:
            large_part = refined & large
            refined = (refined & ~large) | opening(large_part, footprint=kernel)
        result[index] = refined

    return refine_components(result, field, base_candidate)


def grow_instances(
    seeds: np.ndarray,
    probabilities: np.ndarray,
    candidate: InstanceCandidate,
) -> np.ndarray:
    if candidate.grow_radius == 0:
        return seeds
    background = probabilities[:, 0]
    field = probabilities[:, 1]
    boundary = probabilities[:, 2]
    support = (
        (field >= candidate.grow_field_threshold)
        & (field >= candidate.grow_background_ratio * background)
        & (field >= candidate.grow_boundary_ratio * boundary)
    )
    kernel = disk(candidate.grow_radius)
    result = np.empty_like(seeds)
    structure = ndimage.generate_binary_structure(2, 1)
    elevation = background + boundary - field
    for index, sample in enumerate(seeds):
        markers, count = ndimage.label(sample, structure=structure)
        if count == 0:
            result[index] = sample
            continue
        local_support = support[index] & dilation(sample, footprint=kernel)
        grown = watershed(
            elevation[index],
            markers=markers,
            mask=local_support,
            watershed_line=True,
            connectivity=1,
        )
        result[index] = grown > 0
    return result


def make_prediction(
    cache: dict[str, np.ndarray],
    base_candidate: RefinedCandidate,
    candidate: InstanceCandidate,
) -> np.ndarray:
    probabilities = blended_probabilities(cache, base_candidate)
    base_prediction = predict(cache, base_candidate)
    split = split_large_components(
        base_prediction, probabilities, base_candidate, candidate
    )
    return grow_instances(split, probabilities, candidate)


def score(cache, base_candidate, candidate):
    prediction = make_prediction(cache, base_candidate, candidate)
    return metrics(cache["masks"], prediction, exact_objects=False)


def create_record(candidate, result):
    return {**asdict(candidate), **result}


def from_record(item):
    names = {field.name for field in fields(InstanceCandidate)}
    return InstanceCandidate(**{key: item[key] for key in names})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--v2-selection-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.6547743739171523)
    args = parser.parse_args()

    cache = load_cache(args.validation_cache)
    base_candidate = load_v2_candidate(args.v2_selection_report)
    split_candidates = [InstanceCandidate()]
    for minimum in (64, 128, 256, 512, 1024):
        for boundary_ratio in (0.0, 2.0, 2.2, 2.5, 3.0, 3.5):
            for open_radius in (0, 1, 2):
                if boundary_ratio or open_radius:
                    split_candidates.append(
                        InstanceCandidate(minimum, boundary_ratio, open_radius)
                    )
    split_records = [
        create_record(candidate, score(cache, base_candidate, candidate))
        for candidate in split_candidates
    ]
    split_records.sort(
        key=lambda item: (item["object_f1"], item["pixel_iou"]), reverse=True
    )
    split_seeds = [from_record(item) for item in split_records[:8]]

    grown_records = []
    for seed in split_seeds:
        for radius in (1, 2, 3):
            for field_threshold in (0.20, 0.30, 0.40):
                for background_ratio in (0.60, 0.75, 0.90):
                    for boundary_ratio in (1.10, 1.30, 1.50, 1.70):
                        candidate = InstanceCandidate(
                            seed.large_min_pixels,
                            seed.large_boundary_ratio,
                            seed.large_open_radius,
                            radius,
                            field_threshold,
                            background_ratio,
                            boundary_ratio,
                        )
                        grown_records.append(
                            create_record(
                                candidate, score(cache, base_candidate, candidate)
                            )
                        )

    records = split_records + grown_records
    eligible = [item for item in records if item["pixel_iou"] >= args.minimum_iou]
    eligible.sort(
        key=lambda item: (item["object_f1"], item["pixel_iou"]), reverse=True
    )
    selected = from_record(eligible[0])
    exact = metrics(
        cache["masks"],
        make_prediction(cache, base_candidate, selected),
        exact_objects=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "v3_validation_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "selection_protocol": "V3 instance refinement selected on validation only",
        "v2_configuration": asdict(base_candidate),
        "selected_configuration": asdict(selected),
        "selected_validation": exact,
        "nearby_top_ten": eligible[:10],
    }
    with (args.output_dir / "v3_selection_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "nearby_top_ten"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

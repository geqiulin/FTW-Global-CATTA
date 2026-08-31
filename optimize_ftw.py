#!/usr/bin/env python
"""Validation-only optimization and official-style evaluation for FTW.

The model weights are never tuned on the test split.  Validation selects a
test-time ensemble and boundary-aware post-processing configuration; that
frozen configuration is then evaluated once on the test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio.features
import shapely.geometry
import torch
import torch.nn.functional as F
from scipy import ndimage
from shapely import STRtree
from skimage.morphology import closing, disk, opening
from skimage.morphology import remove_small_holes, remove_small_objects
from torch.utils.data import DataLoader

from ftw_tools.training.datasets import FTW
from ftw_tools.training.trainers import CustomSemanticSegmentationTask


@dataclass(frozen=True)
class Candidate:
    ensemble: str
    field_threshold: float
    background_ratio: float
    boundary_ratio: float
    min_object_pixels: int = 0
    max_hole_pixels: int = 0
    close_radius: int = 0
    open_radius: int = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predict_variant(
    model, images, temporal_swap=False, flip_dims=(), resize_factor=1
):
    transformed = images
    if temporal_swap:
        transformed = torch.cat((transformed[:, 4:], transformed[:, :4]), dim=1)
    original_size = transformed.shape[-2:]
    if resize_factor != 1:
        transformed = F.interpolate(
            transformed,
            scale_factor=resize_factor,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    if flip_dims:
        transformed = torch.flip(transformed, dims=flip_dims)
    # Mixed-precision inference is essential for the larger EfficientNet-B5/B7
    # checkpoints on 4 GB GPUs.  Softmax is promoted back to fp32 so cached
    # probabilities and downstream threshold searches remain numerically stable.
    with torch.amp.autocast("cuda", enabled=transformed.is_cuda):
        logits = model(transformed)
    probabilities = torch.softmax(logits.float(), dim=1)
    if flip_dims:
        probabilities = torch.flip(probabilities, dims=flip_dims)
    if resize_factor != 1:
        probabilities = F.interpolate(
            probabilities,
            size=original_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return probabilities


def build_cache(
    checkpoint: Path,
    data_root: Path,
    country: str,
    split: str,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    if cache_path.exists():
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    print(f"Building {split} prediction cache")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    model = task.model.eval().to(device)
    dataset = FTW(
        root=str(data_root),
        countries=[country],
        split=split,
        load_boundaries=False,
        temporal_options="stacked",
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    ensembles = {
        name: []
        for name in (
            "base",
            "spatial4",
            "temporal2",
            "full8",
            "scale2_base",
            "scale2_spatial4",
            "multiscale",
        )
    }
    masks = []
    spatial_flips = ((), (3,), (2,), (2, 3))
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device) / 3000.0
            masks.append(batch["mask"].numpy().astype(np.uint8))
            predictions = {}
            for temporal_swap in (False, True):
                for flip_dims in spatial_flips:
                    key = (temporal_swap, flip_dims)
                    predictions[key] = _predict_variant(
                        model,
                        images,
                        temporal_swap=temporal_swap,
                        flip_dims=flip_dims,
                    )

            base = predictions[(False, ())]
            spatial4 = torch.stack(
                [predictions[(False, dims)] for dims in spatial_flips]
            ).mean(0)
            temporal2 = torch.stack(
                [predictions[(False, ())], predictions[(True, ())]]
            ).mean(0)
            full8 = torch.stack(list(predictions.values())).mean(0)
            scale2_predictions = {
                flip_dims: _predict_variant(
                    model, images, flip_dims=flip_dims, resize_factor=2
                )
                for flip_dims in spatial_flips
            }
            scale2_base = scale2_predictions[()]
            scale2_spatial4 = torch.stack(list(scale2_predictions.values())).mean(0)
            multiscale = (full8 + scale2_spatial4) / 2
            for name, tensor in (
                ("base", base),
                ("spatial4", spatial4),
                ("temporal2", temporal2),
                ("full8", full8),
                ("scale2_base", scale2_base),
                ("scale2_spatial4", scale2_spatial4),
                ("multiscale", multiscale),
            ):
                ensembles[name].append(tensor.cpu().numpy().astype(np.float32))
            print(f"  batch {batch_index}/{len(loader)}")

    payload = {name: np.concatenate(parts) for name, parts in ensembles.items()}
    payload["masks"] = np.concatenate(masks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    del model, task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def decision_mask(probabilities: np.ndarray, candidate: Candidate) -> np.ndarray:
    background = probabilities[:, 0].astype(np.float32)
    field = probabilities[:, 1].astype(np.float32)
    boundary = probabilities[:, 2].astype(np.float32)
    prediction = (
        (field >= candidate.field_threshold)
        & (field >= candidate.background_ratio * background)
        & (field >= candidate.boundary_ratio * boundary)
    )
    if not any(
        (
            candidate.min_object_pixels,
            candidate.max_hole_pixels,
            candidate.close_radius,
            candidate.open_radius,
        )
    ):
        return prediction

    processed = np.empty_like(prediction)
    close_kernel = disk(candidate.close_radius) if candidate.close_radius else None
    open_kernel = disk(candidate.open_radius) if candidate.open_radius else None
    for index, mask in enumerate(prediction):
        result = mask
        if candidate.min_object_pixels:
            result = remove_small_objects(
                result, max_size=candidate.min_object_pixels - 1, connectivity=1
            )
        if close_kernel is not None:
            result = closing(result, footprint=close_kernel)
        if open_kernel is not None:
            result = opening(result, footprint=open_kernel)
        if candidate.max_hole_pixels:
            result = remove_small_holes(
                result, max_size=candidate.max_hole_pixels, connectivity=1
            )
        processed[index] = result
    return processed


def _fast_object_counts(truth: np.ndarray, prediction: np.ndarray):
    structure = ndimage.generate_binary_structure(2, 1)
    true_labels, true_count = ndimage.label(truth == 1, structure=structure)
    pred_labels, pred_count = ndimage.label(prediction == 1, structure=structure)
    if true_count == 0:
        return 0, int(pred_count), 0
    if pred_count == 0:
        return 0, 0, int(true_count)

    true_sizes = np.bincount(true_labels.ravel(), minlength=true_count + 1)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
    overlap = (true_labels > 0) & (pred_labels > 0)
    pair_codes = true_labels[overlap] * (pred_count + 1) + pred_labels[overlap]
    intersections = np.bincount(
        pair_codes, minlength=(true_count + 1) * (pred_count + 1)
    ).reshape(true_count + 1, pred_count + 1)
    true_ids, pred_ids = np.nonzero(intersections[1:, 1:])
    true_ids = true_ids + 1
    pred_ids = pred_ids + 1
    inter = intersections[true_ids, pred_ids]
    union = true_sizes[true_ids] + pred_sizes[pred_ids] - inter
    matched = inter / union > 0.5
    matched_true = np.unique(true_ids[matched]).size
    matched_pred = np.unique(pred_ids[matched]).size
    true_positives = int(matched_true)
    return true_positives, int(pred_count - matched_pred), int(true_count - matched_true)


def _indexed_official_object_counts(truth: np.ndarray, prediction: np.ndarray):
    """Official polygon IoU rule accelerated with a spatial index."""
    true_shapes = [
        shapely.geometry.shape(geometry)
        for geometry, value in rasterio.features.shapes(truth.astype(np.uint8))
        if value == 1
    ]
    pred_shapes = [
        shapely.geometry.shape(geometry)
        for geometry, value in rasterio.features.shapes(prediction.astype(np.uint8))
        if value == 1
    ]
    if not true_shapes:
        return 0, len(pred_shapes), 0
    if not pred_shapes:
        return 0, 0, len(true_shapes)
    tree = STRtree(pred_shapes)
    matched_predictions = set()
    true_positives = 0
    for true_shape in true_shapes:
        matched = None
        for pred_index in tree.query(true_shape):
            pred_index = int(pred_index)
            pred_shape = pred_shapes[pred_index]
            intersection = true_shape.intersection(pred_shape)
            union = true_shape.union(pred_shape)
            if intersection.area / union.area > 0.5:
                matched = pred_index
                break
        if matched is not None:
            true_positives += 1
            matched_predictions.add(matched)
    return (
        true_positives,
        len(pred_shapes) - len(matched_predictions),
        len(true_shapes) - true_positives,
    )


def metrics(masks: np.ndarray, predictions: np.ndarray, exact_objects=False):
    valid = masks != 3
    truth = masks == 1
    pred = predictions.astype(bool)
    tp = int(np.count_nonzero(valid & truth & pred))
    fp = int(np.count_nonzero(valid & ~truth & pred))
    fn = int(np.count_nonzero(valid & truth & ~pred))
    pixel_iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    pixel_precision = tp / (tp + fp) if tp + fp else float("nan")
    pixel_recall = tp / (tp + fn) if tp + fn else float("nan")

    object_tp = object_fp = object_fn = 0
    for true_mask, pred_mask in zip(masks, predictions):
        if exact_objects:
            counts = _indexed_official_object_counts(true_mask, pred_mask)
        else:
            counts = _fast_object_counts(true_mask, pred_mask)
        object_tp += counts[0]
        object_fp += counts[1]
        object_fn += counts[2]
    object_precision = (
        object_tp / (object_tp + object_fp) if object_tp + object_fp else float("nan")
    )
    object_recall = (
        object_tp / (object_tp + object_fn) if object_tp + object_fn else float("nan")
    )
    object_f1 = (
        2 * object_precision * object_recall / (object_precision + object_recall)
        if object_precision + object_recall
        else float("nan")
    )
    return {
        "pixel_iou": pixel_iou,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_tp": tp,
        "pixel_fp": fp,
        "pixel_fn": fn,
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
        "object_tp": object_tp,
        "object_fp": object_fp,
        "object_fn": object_fn,
    }


def score_candidate(cache, candidate: Candidate, exact_objects=False):
    predictions = decision_mask(cache[candidate.ensemble], candidate)
    result = metrics(cache["masks"], predictions, exact_objects=exact_objects)
    return result, predictions


def rank_key(record):
    return (record["object_f1"], record["pixel_iou"])


def _candidate_from_record(record):
    candidate_keys = asdict(Candidate("", 0, 0, 0))
    return Candidate(**{key: record[key] for key in candidate_keys})


def optimize(validation_cache, minimum_balanced_iou: float):
    stage_one = []
    ensemble_names = [name for name in validation_cache if name != "masks"]
    for ensemble in ensemble_names:
        for threshold in (0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            for background_ratio in (0.8, 1.0, 1.2):
                for boundary_ratio in (0.6, 0.8, 1.0, 1.2, 1.5):
                    candidate = Candidate(
                        ensemble, threshold, background_ratio, boundary_ratio
                    )
                    result, _ = score_candidate(validation_cache, candidate)
                    stage_one.append({**asdict(candidate), **result})
    stage_one.sort(key=rank_key, reverse=True)

    stage_two = []
    seeds = []
    seen = set()
    for record in stage_one:
        key = (
            record["ensemble"],
            record["field_threshold"],
            record["background_ratio"],
            record["boundary_ratio"],
        )
        if key not in seen:
            seen.add(key)
            seeds.append(key)
        if len(seeds) == 8:
            break

    for ensemble, threshold, background_ratio, boundary_ratio in seeds:
        for min_pixels in (0, 4, 8, 16, 32, 64, 128):
            for holes in (0, 8, 16, 32, 64):
                for close_radius, open_radius in ((0, 0), (1, 0), (0, 1)):
                    candidate = Candidate(
                        ensemble,
                        threshold,
                        background_ratio,
                        boundary_ratio,
                        min_pixels,
                        holes,
                        close_radius,
                        open_radius,
                    )
                    result, _ = score_candidate(validation_cache, candidate)
                    stage_two.append({**asdict(candidate), **result})
    all_results = stage_one + stage_two
    all_results.sort(key=rank_key, reverse=True)
    best_record = all_results[0]
    balanced_records = [
        record for record in all_results if record["pixel_iou"] >= minimum_balanced_iou
    ]
    balanced_records.sort(key=rank_key, reverse=True)
    return (
        _candidate_from_record(best_record),
        _candidate_from_record(balanced_records[0]),
        all_results,
    )


def write_search_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_submission_csv(path: Path, country: str, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Country",
        "PixelIoU",
        "PixelPrecision",
        "PixelRecall",
        "ObjectPrecision",
        "ObjectRecall",
        "ObjectF1",
    ]
    row = {
        "Country": country,
        "PixelIoU": result["pixel_iou"],
        "PixelPrecision": result["pixel_precision"],
        "PixelRecall": result["pixel_recall"],
        "ObjectPrecision": result["object_precision"],
        "ObjectRecall": result["object_recall"],
        "ObjectF1": result["object_f1"],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--country", default="vietnam")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validation_cache = build_cache(
        checkpoint,
        data_root,
        args.country,
        "val",
        args.output_dir / "cache_val.npz",
        device,
        args.batch_size,
        args.num_workers,
    )
    test_cache = build_cache(
        checkpoint,
        data_root,
        args.country,
        "test",
        args.output_dir / "cache_test.npz",
        device,
        args.batch_size,
        args.num_workers,
    )

    baseline = Candidate("base", 0.0, 1.0, 1.0)
    baseline_val, _ = score_candidate(validation_cache, baseline, exact_objects=True)
    baseline_test, _ = score_candidate(test_cache, baseline, exact_objects=True)
    best, balanced, search_records = optimize(
        validation_cache, minimum_balanced_iou=baseline_val["pixel_iou"]
    )
    write_search_csv(args.output_dir / "validation_search.csv", search_records)
    optimized_val, _ = score_candidate(validation_cache, best, exact_objects=True)
    optimized_test, optimized_predictions = score_candidate(
        test_cache, best, exact_objects=True
    )
    balanced_val, _ = score_candidate(
        validation_cache, balanced, exact_objects=True
    )
    balanced_test, balanced_predictions = score_candidate(
        test_cache, balanced, exact_objects=True
    )
    np.savez_compressed(
        args.output_dir / "predictions_vietnam_test.npz",
        predictions=optimized_predictions.astype(np.uint8),
        masks=test_cache["masks"].astype(np.uint8),
    )
    write_submission_csv(
        args.output_dir / "ftw_leaderboard_metrics_primary_f1.csv",
        args.country,
        optimized_test,
    )
    write_submission_csv(
        args.output_dir / "ftw_leaderboard_metrics.csv", args.country, balanced_test
    )
    np.savez_compressed(
        args.output_dir / "predictions_vietnam_test_balanced.npz",
        predictions=balanced_predictions.astype(np.uint8),
        masks=test_cache["masks"].astype(np.uint8),
    )

    report = {
        "benchmark": "Fields of the World",
        "country": args.country,
        "selection_protocol": "validation-only selection; single frozen test evaluation",
        "ranking_metric": "object_f1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device": str(device),
        "torch_version": torch.__version__,
        "baseline_configuration": asdict(baseline),
        "selected_configuration": asdict(best),
        "balanced_configuration": asdict(balanced),
        "baseline_validation": baseline_val,
        "optimized_validation": optimized_val,
        "baseline_test": baseline_test,
        "optimized_test": optimized_test,
        "balanced_validation": balanced_val,
        "balanced_test": balanced_test,
        "test_object_f1_absolute_gain": optimized_test["object_f1"]
        - baseline_test["object_f1"],
        "test_pixel_iou_absolute_gain": optimized_test["pixel_iou"]
        - baseline_test["pixel_iou"],
        "recommended_submission": "balanced",
        "balanced_test_object_f1_absolute_gain": balanced_test["object_f1"]
        - baseline_test["object_f1"],
        "balanced_test_pixel_iou_absolute_gain": balanced_test["pixel_iou"]
        - baseline_test["pixel_iou"],
        "runtime_seconds": time.time() - started,
    }
    with (args.output_dir / "optimization_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

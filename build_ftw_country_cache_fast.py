#!/usr/bin/env python
"""Build a compact spatial-TTA FTW cache for multi-country optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ftw_tools.training.datasets import FTW
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from optimize_ftw import _predict_variant, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--include-scale2",
        action="store_true",
        help="Also cache one 2x-resolution prediction (higher cost).",
    )
    args = parser.parse_args()

    if args.output.exists():
        with np.load(args.output) as cached:
            print(
                f"CACHE_EXISTS country={args.country} split={args.split} "
                f"samples={len(cached['masks'])}"
            )
        return

    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    model = task.model.eval().to(device)
    dataset = FTW(
        root=str(data_root),
        countries=[args.country],
        split=args.split,
        load_boundaries=False,
        temporal_options="stacked",
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    arrays = {"base": [], "spatial4": []}
    if args.include_scale2:
        arrays["scale2_base"] = []
    masks = []
    flips = ((), (3,), (2,), (2, 3))
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device) / 3000.0
            masks.append(batch["mask"].numpy().astype(np.uint8))
            predictions = [
                _predict_variant(model, images, flip_dims=dims)
                for dims in flips
            ]
            arrays["base"].append(predictions[0].cpu().numpy().astype(np.float16))
            spatial4 = torch.stack(predictions).mean(0)
            arrays["spatial4"].append(
                spatial4.cpu().numpy().astype(np.float16)
            )
            if args.include_scale2:
                scale2 = _predict_variant(model, images, resize_factor=2)
                arrays["scale2_base"].append(
                    scale2.cpu().numpy().astype(np.float16)
                )
            print(f"  batch {batch_index}/{len(loader)}", flush=True)

    payload = {name: np.concatenate(parts) for name, parts in arrays.items()}
    payload["masks"] = np.concatenate(masks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata = {
        "country": args.country,
        "split": args.split,
        "samples": len(payload["masks"]),
        "ensembles": list(arrays),
        "probability_dtype": "float16",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device": str(device),
        "selection_use": "validation" if args.split == "val" else "frozen test",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"CACHE_COMPLETE country={args.country} split={args.split} "
        f"samples={len(payload['masks'])} device={device}"
    )


if __name__ == "__main__":
    main()

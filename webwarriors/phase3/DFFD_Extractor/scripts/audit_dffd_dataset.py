#!/usr/bin/env python3
import os
import json
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/deepfake")
OUT = Path("/opt/dffd/output")
SPLITS = [
    "FaceAPP_Train",
    "FaceAPP_Val",
    "FFHQ_Train",
    "FFHQ_Val",
]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SAMPLE_LIMIT = 8

def list_images(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    )

def index_by_stem(paths):
    result = {}
    duplicates = []
    for p in paths:
        key = p.stem
        if key in result:
            duplicates.append(key)
        else:
            result[key] = p
    return result, duplicates

def image_info(path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    arr = cv2.imread(str(path), flag)
    if arr is None:
        return {"readable": False}

    result = {
        "readable": True,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "file_bytes": path.stat().st_size,
    }

    if grayscale:
        vals, counts = np.unique(arr, return_counts=True)
        result.update({
            "gray_min": int(arr.min()),
            "gray_max": int(arr.max()),
            "gray_unique_count": int(len(vals)),
            "gray_exact_0_ratio": float(np.mean(arr == 0)),
            "gray_exact_255_ratio": float(np.mean(arr == 255)),
            "gray_intermediate_ratio": float(np.mean((arr > 0) & (arr < 255))),
            "gray_most_common": [
                {"value": int(vals[i]), "count": int(counts[i])}
                for i in np.argsort(counts)[-8:][::-1]
            ],
        })
    return result

def audit_split(split):
    base = ROOT / split
    image_dir = base / "image"
    mask_dir = base / "mask"

    images = list_images(image_dir)
    masks = list_images(mask_dir)

    image_map, image_dups = index_by_stem(images)
    mask_map, mask_dups = index_by_stem(masks)

    image_stems = set(image_map)
    mask_stems = set(mask_map)
    paired = sorted(image_stems & mask_stems)
    images_without_masks = sorted(image_stems - mask_stems)
    masks_without_images = sorted(mask_stems - image_stems)

    image_dims = Counter()
    image_exts = Counter()
    unreadable_images = []
    image_size_samples = []

    for p in images:
        image_exts[p.suffix.lower()] += 1
        info = image_info(p)
        if not info["readable"]:
            unreadable_images.append(p.name)
            continue
        image_dims[f'{info["width"]}x{info["height"]}'] += 1
        if len(image_size_samples) < SAMPLE_LIMIT:
            image_size_samples.append({"file": p.name, **info})

    mask_dims = Counter()
    mask_shapes = Counter()
    unreadable_masks = []
    mask_samples = []
    mask_gray_maxima = Counter()
    mask_exact255_ratios = []

    for p in masks:
        info = image_info(p)
        if not info["readable"]:
            unreadable_masks.append(p.name)
            continue
        mask_dims[f'{info["width"]}x{info["height"]}'] += 1
        mask_shapes[str(info["shape"])] += 1
        gray = image_info(p, grayscale=True)
        mask_gray_maxima[str(gray["gray_max"])] += 1
        mask_exact255_ratios.append(gray["gray_exact_255_ratio"])
        if len(mask_samples) < SAMPLE_LIMIT:
            mask_samples.append({"file": p.name, **info, "grayscale": gray})

    mismatched_dimensions = []
    for stem in paired:
        ip = image_map[stem]
        mp = mask_map[stem]
        ii = image_info(ip)
        mi = image_info(mp)
        if ii.get("readable") and mi.get("readable"):
            if (ii["width"], ii["height"]) != (mi["width"], mi["height"]):
                mismatched_dimensions.append({
                    "stem": stem,
                    "image": ip.name,
                    "mask": mp.name,
                    "image_size": [ii["width"], ii["height"]],
                    "mask_size": [mi["width"], mi["height"]],
                })

    return {
        "split": split,
        "directories": {
            "split_exists": base.is_dir(),
            "image_dir_exists": image_dir.is_dir(),
            "mask_dir_exists": mask_dir.is_dir(),
        },
        "counts": {
            "images": len(images),
            "masks": len(masks),
            "unique_image_stems": len(image_stems),
            "unique_mask_stems": len(mask_stems),
            "matching_stems": len(paired),
            "images_without_masks": len(images_without_masks),
            "masks_without_images": len(masks_without_images),
            "duplicate_image_stems": len(image_dups),
            "duplicate_mask_stems": len(mask_dups),
            "unreadable_images": len(unreadable_images),
            "unreadable_masks": len(unreadable_masks),
            "paired_dimension_mismatches": len(mismatched_dimensions),
        },
        "image_extensions": dict(image_exts),
        "image_dimensions": dict(image_dims),
        "mask_dimensions": dict(mask_dims),
        "mask_shapes": dict(mask_shapes),
        "mask_grayscale_maxima_distribution": dict(mask_gray_maxima),
        "mask_exact255_ratio_summary": (
            {
                "min": float(min(mask_exact255_ratios)),
                "median": float(np.median(mask_exact255_ratios)),
                "max": float(max(mask_exact255_ratios)),
            } if mask_exact255_ratios else None
        ),
        "examples": {
            "image_files": [p.name for p in images[:SAMPLE_LIMIT]],
            "mask_files": [p.name for p in masks[:SAMPLE_LIMIT]],
            "matching_stems": paired[:SAMPLE_LIMIT],
            "images_without_masks": images_without_masks[:SAMPLE_LIMIT],
            "masks_without_images": masks_without_images[:SAMPLE_LIMIT],
            "duplicate_image_stems": image_dups[:SAMPLE_LIMIT],
            "duplicate_mask_stems": mask_dups[:SAMPLE_LIMIT],
            "unreadable_images": unreadable_images[:SAMPLE_LIMIT],
            "unreadable_masks": unreadable_masks[:SAMPLE_LIMIT],
            "dimension_mismatches": mismatched_dimensions[:SAMPLE_LIMIT],
            "image_samples": image_size_samples,
            "mask_samples": mask_samples,
        },
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_root": str(ROOT),
        "audit_is_read_only": True,
        "splits": [audit_split(s) for s in SPLITS],
    }

    report_path = OUT / "dffd_dataset_audit.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== DFFD DATASET AUDIT ==========")
    print("Dataset:", ROOT)
    for s in report["splits"]:
        print(f"\n--- {s['split']} ---")
        print("Directories:", s["directories"])
        for key, value in s["counts"].items():
            print(f"{key}: {value}")
        print("Image dimensions:", s["image_dimensions"])
        print("Mask dimensions:", s["mask_dimensions"])
        print("Mask shapes:", s["mask_shapes"])
        print("Mask grayscale max distribution:",
              s["mask_grayscale_maxima_distribution"])
        print("Sample image names:", s["examples"]["image_files"][:5])
        print("Sample mask names:", s["examples"]["mask_files"][:5])
        print("Sample matching stems:", s["examples"]["matching_stems"][:5])
        print("Sample images without masks:",
              s["examples"]["images_without_masks"][:5])

    print("\nFull JSON report:", report_path, flush=True)

if __name__ == "__main__":
    main()

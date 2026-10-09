import os
import csv
import argparse
from PIL import Image


def calculate_metrics(gt_path, pred_path):
    """
    Calculate IoU and pixel F1 for one image.

    Ground truth:
        FaceAPP/DFFD mask encoding
        gray / 255 > 0.1
        equivalent to gray > 25.5

    Prediction:
        MFLM output is binary 0/255
        foreground = pixel > 127
    """

    gt = Image.open(gt_path).convert("L")
    pred = Image.open(pred_path).convert("L")

    if gt.size != pred.size:
        raise ValueError(
            f"Size mismatch: "
            f"GT={gt.size}, Prediction={pred.size}"
        )

    gt_pixels = list(gt.getdata())
    pred_pixels = list(pred.getdata())

    # Ground-truth mask
    gt_bin = [
        value > 25.5
        for value in gt_pixels
    ]

    # MFLM prediction mask
    pred_bin = [
        value > 127
        for value in pred_pixels
    ]

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for gt_value, pred_value in zip(
        gt_bin,
        pred_bin
    ):
        if gt_value and pred_value:
            tp += 1

        elif not gt_value and pred_value:
            fp += 1

        elif gt_value and not pred_value:
            fn += 1

        else:
            tn += 1

    # IoU
    union = tp + fp + fn

    if union > 0:
        iou = tp / union
    else:
        iou = 1.0

    # Pixel F1
    denominator = (
        2 * tp +
        fp +
        fn
    )

    if denominator > 0:
        pixel_f1 = (
            2 * tp
        ) / denominator
    else:
        pixel_f1 = 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_foreground": sum(gt_bin),
        "pred_foreground": sum(pred_bin),
        "iou": iou,
        "pixel_f1": pixel_f1,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Calculate FakeShield localization metrics."
    )

    # ---------------------------------------------------------
    # WebWarriors test-bed
    # ---------------------------------------------------------

    parser.add_argument(
        "--dataset-dir",
        default=(
            "./webwarriors/baseline/dataset"
        ),
    )

    # ---------------------------------------------------------
    # MFLM predictions
    # ---------------------------------------------------------

    parser.add_argument(
        "--pred-dir",
        default=(
            "./webwarriors/baseline/results/mflm"
        ),
    )

    # ---------------------------------------------------------
    # Metrics output
    # ---------------------------------------------------------

    parser.add_argument(
        "--output",
        default=(f"./webwarriors/baseline/results/metrics.csv"),
    )

    args = parser.parse_args()

    image_dir = os.path.join(
        args.dataset_dir,
        "image"
    )

    gt_dir = os.path.join(
        args.dataset_dir,
        "mask"
    )

    pred_dir = args.pred_dir

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    print()
    print("========================================")
    print("Localization Metrics")
    print("========================================")
    print()

    print("Test:")
    print(args.dataset_dir)

    print()
    print("Image directory:")
    print(image_dir)

    print()
    print("Ground-truth directory:")
    print(gt_dir)

    print()
    print("Prediction directory:")
    print(pred_dir)

    print()

    # ---------------------------------------------------------
    # Read ONLY the images belonging to our test-bed
    # ---------------------------------------------------------

    image_files = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    )

    print(
        "Test images:",
        len(image_files)
    )

    # ---------------------------------------------------------
    # Read GT masks belonging to test-bed
    # ---------------------------------------------------------

    gt_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(gt_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    }

    print(
        "Test GT masks:",
        len(gt_files)
    )

    print()

    results = []

    missing_gt = 0
    missing_prediction = 0
    failed = 0

    # ---------------------------------------------------------
    # Evaluate ONLY test images
    # ---------------------------------------------------------

    for image_file in image_files:

        stem = os.path.splitext(
            image_file
        )[0]

        # -------------------------------
        # Ground truth
        # -------------------------------

        if stem not in gt_files:

            print(
                f"SKIP {image_file}: "
                f"GT mask not found"
            )

            missing_gt += 1
            continue

        gt_file = gt_files[stem]

        gt_path = os.path.join(
            gt_dir,
            gt_file
        )

        # -------------------------------
        # Prediction
        # -------------------------------

        pred_candidates = [
            stem + ".png",
            stem + ".jpg",
            stem + ".jpeg",
        ]

        pred_path = None

        for candidate in pred_candidates:

            candidate_path = os.path.join(
                pred_dir,
                candidate
            )

            if os.path.exists(
                candidate_path
            ):
                pred_path = candidate_path
                break

        if pred_path is None:

            print(
                f"SKIP {image_file}: "
                f"prediction not found"
            )

            missing_prediction += 1
            continue

        # -------------------------------
        # Calculate
        # -------------------------------

        try:

            metrics = calculate_metrics(
                gt_path,
                pred_path
            )

            result = {
                "filename": image_file,
                **metrics,
            }

            results.append(result)

            print(
                f"[{len(results):03d}] "
                f"{image_file:<30} "
                f"IoU={metrics['iou']:.6f} "
                f"F1={metrics['pixel_f1']:.6f}"
            )

        except Exception as e:

            print(
                f"ERROR {image_file}: "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

    # ---------------------------------------------------------
    # Save per-image metrics
    # ---------------------------------------------------------

    fieldnames = [
        "filename",
        "tp",
        "fp",
        "fn",
        "tn",
        "gt_foreground",
        "pred_foreground",
        "iou",
        "pixel_f1",
    ]

    with open(
        args.output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------

    if results:

        mean_iou = (
            sum(
                r["iou"]
                for r in results
            )
            / len(results)
        )

        mean_f1 = (
            sum(
                r["pixel_f1"]
                for r in results
            )
            / len(results)
        )

    else:

        mean_iou = 0.0
        mean_f1 = 0.0

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("FINAL RESULTS")
    print("========================================")

    print(
        "Test images:",
        len(image_files)
    )

    print(
        "Test GT masks:",
        len(gt_files)
    )

    print(
        "Evaluated:",
        len(results)
    )

    print(
        "Missing GT:",
        missing_gt
    )

    print(
        "Missing predictions:",
        missing_prediction
    )

    print(
        "Failed:",
        failed
    )

    print()

    print(
        f"Mean IoU:      {mean_iou:.6f}"
    )

    print(
        f"Mean Pixel F1: {mean_f1:.6f}"
    )

    print()

    print(
        "Per-image results:"
    )

    print(
        args.output
    )

    print("========================================")


if __name__ == "__main__":
    main()
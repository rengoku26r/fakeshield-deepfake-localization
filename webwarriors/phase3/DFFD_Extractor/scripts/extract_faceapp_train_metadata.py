#!/usr/bin/env python3
import argparse
import json
import os
import sys
sys.path.insert(0, "/opt/dffd/build_work")
sys.path.insert(1, "/opt/dffd/extract_faces")
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from retinaface import RetinaFace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-prefix", default="./model/R50")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    # Resume safely: only skip records with a completed status.
    completed = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("status") in {"ok", "no_face", "unreadable"}:
                        completed.add(record["image"])
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"Input directory: {input_dir}", flush=True)
    print(f"Images discovered: {len(files)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print("Initializing RetinaFace...", flush=True)

    detector = RetinaFace("/opt/dffd/extract_faces/model/R50", 0, -1, "net3")
    print("Detector initialized.", flush=True)

    started = time.time()
    newly_done = 0
    failures = []

    with output_path.open("a", encoding="utf-8", buffering=1) as out:
        for index, path in enumerate(files, start=1):
            name = path.name
            if name in completed:
                continue

            record = {
                "image": name,
                "width": None,
                "height": None,
                "status": "error",
                "detector": "RetinaFace R50 net3",
                "threshold": args.threshold,
                "boxes_xyxy_score": [],
                "landmarks_5x2": [],
                "selected_face_index": None,
                "selected_box_xyxy_score": None,
                "selected_landmarks_5x2": None,
            }

            try:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    record["status"] = "unreadable"
                    record["error"] = "cv2.imread returned None"
                    failures.append({"image": name, "reason": "unreadable"})
                else:
                    height, width = image.shape[:2]
                    record["width"] = int(width)
                    record["height"] = int(height)

                    # Match the original DFFD extraction script's scaling.
                    target_size, max_size = 512, 990
                    min_dim = min(height, width)
                    max_dim = max(height, width)
                    scale = float(target_size) / float(min_dim)
                    if round(scale * max_dim) > max_size:
                        scale = float(max_size) / float(max_dim)

                    faces, landmarks = detector.detect(
                        image,
                        args.threshold,
                        scales=[scale],
                        do_flip=False,
                    )

                    if faces is None or len(faces) == 0:
                        record["status"] = "no_face"
                        record["error"] = "No face detected above threshold"
                        failures.append({
                            "image": name,
                            "reason": "no_face",
                        })
                    else:
                        faces_array = np.asarray(faces, dtype=float)
                        landmarks_array = (
                            np.asarray(landmarks, dtype=float)
                            if landmarks is not None else None
                        )

                        record["boxes_xyxy_score"] = faces_array.tolist()
                        record["landmarks_5x2"] = (
                            landmarks_array.tolist()
                            if landmarks_array is not None else []
                        )

                        # Select the largest bounding box by area.
                        widths = np.maximum(
                            0, faces_array[:, 2] - faces_array[:, 0]
                        )
                        heights = np.maximum(
                            0, faces_array[:, 3] - faces_array[:, 1]
                        )
                        selected = int(np.argmax(widths * heights))

                        record["selected_face_index"] = selected
                        record["selected_box_xyxy_score"] = (
                            faces_array[selected].tolist()
                        )
                        if (
                            landmarks_array is not None
                            and len(landmarks_array) > selected
                        ):
                            record["selected_landmarks_5x2"] = (
                                landmarks_array[selected].tolist()
                            )

                        record["status"] = "ok"

            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                failures.append({
                    "image": name,
                    "reason": repr(exc),
                })
                print(f"\nERROR {name}: {exc}", file=sys.stderr, flush=True)

            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            newly_done += 1

            if newly_done % args.log_every == 0:
                elapsed = time.time() - started
                print(
                    f"Processed this run: {newly_done}; "
                    f"file: {name}; elapsed: {elapsed:.1f}s",
                    flush=True,
                )

    # Re-read final output to report cumulative counts.
    totals = {"ok": 0, "no_face": 0, "unreadable": 0, "error": 0}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                status = json.loads(line).get("status")
                if status in totals:
                    totals[status] += 1
            except json.JSONDecodeError:
                pass

    failures_path = output_path.with_name("faceapp_train_failures.json")
    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    summary_path = output_path.with_name("faceapp_train_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "input_dir": str(input_dir),
            "output_jsonl": str(output_path),
            "images_discovered_this_run": len(files),
            "new_records_this_run": newly_done,
            "status_counts_cumulative": totals,
            "elapsed_seconds_this_run": round(time.time() - started, 2),
        }, f, indent=2)

    print("\n========== EXTRACTION SUMMARY ==========", flush=True)
    print(json.dumps(totals, indent=2), flush=True)
    print(f"Metadata: {output_path}", flush=True)
    print(f"Failures: {failures_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

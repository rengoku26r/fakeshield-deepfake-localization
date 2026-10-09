from pathlib import Path
import shutil

DATASET_ROOT = Path(
    "./dataset/deepfake"
)

OUTPUT_ROOT = Path(
    "./webwarriors/baseline/dataset"
)

#   20   -> first 20 images
#   50   -> first 50 images
#   100  -> first 100 images
#   500  -> first 500 images
#   None -> entire FaceAPP_Val
TARGET_SIZE = 20


# ============================================================
# Dataset
# ============================================================

IMAGE_DIR = DATASET_ROOT / "FaceAPP_Val" / "image"
MASK_DIR = DATASET_ROOT / "FaceAPP_Val" / "mask"

OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "image"
OUTPUT_MASK_DIR = OUTPUT_ROOT / "mask"


def main():

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Find valid FaceAPP image/mask pairs
    # --------------------------------------------------------

    pairs = []

    for image_path in sorted(IMAGE_DIR.iterdir()):

        if not image_path.is_file():
            continue

        mask_path = MASK_DIR / image_path.name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    # --------------------------------------------------------
    # Decide target number
    # --------------------------------------------------------

    if TARGET_SIZE is None:
        target_size = len(pairs)
    else:
        target_size = min(TARGET_SIZE, len(pairs))

    selected_pairs = pairs[:target_size]

    # --------------------------------------------------------
    # Copy only files that are not already present
    # --------------------------------------------------------

    copied = 0
    already_present = 0

    for image_path, mask_path in selected_pairs:

        output_image = OUTPUT_IMAGE_DIR / image_path.name
        output_mask = OUTPUT_MASK_DIR / mask_path.name

        # If both already exist, do not touch them again.
        if output_image.exists() and output_mask.exists():
            already_present += 1
            continue

        shutil.copy2(image_path, output_image)
        shutil.copy2(mask_path, output_mask)

        copied += 1

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    current_images = sorted(
        p for p in OUTPUT_IMAGE_DIR.iterdir()
        if p.is_file()
    )

    current_masks = sorted(
        p for p in OUTPUT_MASK_DIR.iterdir()
        if p.is_file()
    )

    print("==============================================")
    print("DeepFake Test")
    print("==============================================")
    print(f"Target size       : {target_size}")
    print(f"Available pairs   : {len(pairs)}")
    print(f"Copied this run   : {copied}")
    print(f"Already present   : {already_present}")
    print(f"Images in subset  : {len(current_images)}")
    print(f"Masks in subset   : {len(current_masks)}")
    print(f"Output directory  : {OUTPUT_ROOT}")
    print("==============================================")

if __name__ == "__main__":
    main()

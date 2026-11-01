import cv2
import numpy as np

image_path = "/data/deepfake/FaceAPP_Train/image/F_FAP0_00000-1.png"
mask_path = "/data/deepfake/FaceAPP_Train/mask/F_FAP0_00000-1.png"
out_path = "/opt/dffd/output/F_FAP0_00000-1_threshold25_overlay.png"

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError("Image or mask could not be loaded")
if image.shape[:2] != mask.shape[:2]:
    raise ValueError(f"Dimension mismatch: {image.shape} vs {mask.shape}")

# Exact convention documented in calculate_metrics.py:
# grayscale ground-truth foreground = pixel > 25.5
binary = mask > 25.5
overlay = image.copy()

green = np.zeros_like(image)
green[:, :, 1] = 255
overlay[binary] = (
    0.65 * overlay[binary] + 0.35 * green[binary]
).astype(np.uint8)

contours, _ = cv2.findContours(
    binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

# RetinaFace coordinates from the successful single-image run.
cv2.rectangle(overlay, (39, 9), (252, 298), (255, 0, 0), 2)
landmarks = [
    (99, 108), (201, 109), (154, 174), (98, 207), (202, 207)
]
for x, y in landmarks:
    cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)

print("Image shape:", image.shape)
print("Mask grayscale range:", int(mask.min()), int(mask.max()))
print("Foreground pixels:", int(binary.sum()), "/", binary.size)
print("Foreground ratio:", float(binary.mean()))
print("Foreground bbox:", cv2.boundingRect(binary.astype(np.uint8)))

if not cv2.imwrite(out_path, overlay):
    raise RuntimeError("Failed to save overlay")
print("Saved:", out_path)

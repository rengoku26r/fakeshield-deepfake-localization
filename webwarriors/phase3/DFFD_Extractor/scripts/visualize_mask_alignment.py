import os
import cv2
import numpy as np

image_path = "/data/deepfake/FaceAPP_Train/image/F_FAP0_00000-1.png"
mask_path  = "/data/deepfake/FaceAPP_Train/mask/F_FAP0_00000-1.png"
output_dir = "/opt/dffd/output"

os.makedirs(output_dir, exist_ok=True)

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

if image is None:
    raise FileNotFoundError(f"Image not found/readable: {image_path}")
if mask is None:
    raise FileNotFoundError(f"Mask not found/readable: {mask_path}")

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask unique values (up to 20):", np.unique(mask)[:20])

if image.shape[:2] != mask.shape[:2]:
    raise ValueError(
        "Image and mask dimensions differ. Do not overlay until alignment is resolved."
    )

# Convert the mask to a single channel for visualization.
if mask.ndim == 3:
    mask_gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
else:
    mask_gray = mask

# Use Otsu thresholding for a diagnostic boundary; this does not alter the source mask.
mask_u8 = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary = cv2.threshold(
    mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

overlay = image.copy()

# Shade the mask region lightly and draw its boundary.
tint = np.zeros_like(image)
tint[:, :, 1] = 255  # green
region = binary > 0
overlay[region] = (
    0.65 * overlay[region] + 0.35 * tint[region]
).astype(np.uint8)
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

# Detector results from the successful single-image test.
x1, y1, x2, y2 = 39.0592, 8.9445, 251.5217, 297.8221
landmarks = np.array([
    [98.51371, 107.874664],
    [201.44693, 108.53776],
    [154.38234, 173.73723],
    [97.85765, 207.30365],
    [201.87064, 206.89507],
])

h, w = image.shape[:2]
x1i, x2i = np.clip([round(x1), round(x2)], 0, w - 1)
y1i, y2i = np.clip([round(y1), round(y2)], 0, h - 1)
cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (255, 0, 0), 2)

for i, (x, y) in enumerate(landmarks, start=1):
    cv2.circle(overlay, (round(x), round(y)), 4, (0, 0, 255), -1)
    cv2.putText(
        overlay, str(i), (round(x) + 4, round(y) - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
    )

out_path = os.path.join(output_dir, "F_FAP0_00000-1_alignment.png")
if not cv2.imwrite(out_path, overlay):
    raise RuntimeError(f"Failed to save overlay: {out_path}")

print("Mask foreground pixel ratio:", float(region.mean()))
print("Mask contours:", len(contours))
print("Overlay saved:", out_path)

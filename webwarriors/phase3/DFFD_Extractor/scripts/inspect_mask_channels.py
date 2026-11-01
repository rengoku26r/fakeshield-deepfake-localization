import cv2
import numpy as np

src = "/data/deepfake/FaceAPP_Train/mask/F_FAP0_00000-1.png"
out = "/opt/dffd/output/F_FAP0_00000-1_channel_diagnostic.png"

print("Reading mask...", flush=True)
mask = cv2.imread(src, cv2.IMREAD_COLOR)
if mask is None:
    raise FileNotFoundError(src)

print("Shape:", mask.shape, flush=True)

b, g, r = cv2.split(mask)
gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

def enhance(channel):
    lo, hi = np.percentile(channel, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(channel)
    return np.clip(
        (channel.astype(np.float32) - lo) * 255 / (hi - lo),
        0, 255
    ).astype(np.uint8)

panels = [
    ("Red", r),
    ("Green", g),
    ("Blue", b),
    ("Grayscale", gray),
]
tiles = []
for name, channel in panels:
    print(
        f"{name}: min={channel.min()}, max={channel.max()}, "
        f"mean={channel.mean():.3f}, unique={len(np.unique(channel))}",
        flush=True
    )
    tile = cv2.cvtColor(enhance(channel), cv2.COLOR_GRAY2BGR)
    cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)

sheet = np.vstack([
    np.hstack([tiles[0], tiles[1]]),
    np.hstack([tiles[2], tiles[3]])
])

if not cv2.imwrite(out, sheet):
    raise RuntimeError("Could not save diagnostic image")

print("SUCCESS — diagnostic saved to:", out, flush=True)

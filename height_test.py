import cv2
import numpy as np
import torch

from PIL import Image
from transformers import pipeline


# ============================================================
# 1. DEVICE
# ============================================================

device = 0 if torch.cuda.is_available() else -1

print("Loading metric depth model...")

if device == -1:
    print("Using CPU")
else:
    print("Using CUDA GPU")


# ============================================================
# 2. LOAD METRIC DEPTH MODEL
# ============================================================

depth_model = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    device=device
)

print("Metric depth model loaded!")


# ============================================================
# 3. LOAD IMAGE
# ============================================================

image_path = "test.jpg"

image = Image.open(image_path).convert("RGB")

print("Image loaded!")


# ============================================================
# 4. RUN DEPTH ESTIMATION
# ============================================================

result = depth_model(image)

print("Depth estimation complete!")


# ============================================================
# 5. GET DEPTH MAP
# ============================================================

depth = result["predicted_depth"]

depth = depth.squeeze().cpu().numpy()


print("Depth map shape:", depth.shape)

print(
    "Minimum depth:",
    np.min(depth)
)

print(
    "Maximum depth:",
    np.max(depth)
)

print(
    "Average depth:",
    np.mean(depth)
)


# ============================================================
# 6. RESIZE DEPTH TO IMAGE SIZE
# ============================================================

image_cv = cv2.imread(image_path)

height, width = image_cv.shape[:2]

depth = cv2.resize(
    depth,
    (width, height)
)


# ============================================================
# 7. SHOW DEPTH AT CENTER
# ============================================================

center_x = width // 2
center_y = height // 2

center_depth = depth[
    center_y,
    center_x
]


print()
print(
    f"Center depth: {center_depth:.2f} m"
)


# ============================================================
# 8. CREATE DEPTH VISUALIZATION
# ============================================================

depth_display = cv2.normalize(
    depth,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_display = depth_display.astype(
    np.uint8
)

depth_colored = cv2.applyColorMap(
    depth_display,
    cv2.COLORMAP_PLASMA
)


# ============================================================
# 9. DISPLAY
# ============================================================

cv2.imshow(
    "Metric Depth",
    depth_colored
)

print()
print("Press Q to quit.")


while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()
import os
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import pipeline


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_IMAGE = "test.jpg"

OUTPUT_FOLDER = "output"


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. DEVICE
# ============================================================

device = 0 if torch.cuda.is_available() else -1

if device == -1:
    print("Using CPU")

else:
    print("Using CUDA GPU")


# ============================================================
# 4. LOAD MODEL
# ============================================================

print()
print("Loading Depth Anything V2...")
print("Model: Metric Outdoor Small")
print()

depth_model = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    device=device
)

print("Model loaded successfully!")


# ============================================================
# 5. LOAD IMAGE
# ============================================================

if not os.path.exists(INPUT_IMAGE):

    print(
        f"ERROR: Could not find {INPUT_IMAGE}"
    )

    exit()


image = Image.open(
    INPUT_IMAGE
).convert("RGB")


print(
    f"Image loaded: {INPUT_IMAGE}"
)


# ============================================================
# 6. RUN DEPTH ESTIMATION
# ============================================================

print()
print("Generating depth map...")
print("This may take some time on CPU.")
print()

result = depth_model(image)

print("Depth estimation complete!")


# ============================================================
# 7. EXTRACT DEPTH
# ============================================================

depth = result["predicted_depth"]

depth = depth.squeeze().cpu().numpy()


print()
print("Depth information:")
print(
    f"Minimum: {np.min(depth):.3f}"
)

print(
    f"Maximum: {np.max(depth):.3f}"
)

print(
    f"Mean:    {np.mean(depth):.3f}"
)


# ============================================================
# 8. RESIZE DEPTH TO ORIGINAL IMAGE
# ============================================================

original_image = cv2.imread(
    INPUT_IMAGE
)

height, width = original_image.shape[:2]

depth = cv2.resize(
    depth,
    (width, height),
    interpolation=cv2.INTER_CUBIC
)


# ============================================================
# 9. SAVE NUMERICAL DEPTH MAP
# ============================================================

depth_path = os.path.join(
    OUTPUT_FOLDER,
    "depth_map.npy"
)

np.save(
    depth_path,
    depth
)

print()
print(
    f"Depth data saved to: {depth_path}"
)


# ============================================================
# 10. CREATE RELATIVE DSM
# ============================================================

# Normalize depth to 0 → 1

depth_min = np.min(depth)

depth_max = np.max(depth)

if depth_max - depth_min == 0:

    relative_dsm = np.zeros_like(
        depth
    )

else:

    relative_dsm = (
    depth_max - depth
) / (
    depth_max - depth_min
)


# ============================================================
# 11. SAVE NUMERICAL rDSM
# ============================================================

rdsm_path = os.path.join(
    OUTPUT_FOLDER,
    "rDSM.npy"
)

np.save(
    rdsm_path,
    relative_dsm
)

print(
    f"rDSM data saved to: {rdsm_path}"
)


# ============================================================
# 12. CREATE VISUAL rDSM
# ============================================================

rdsm_8bit = (
    relative_dsm * 255
).astype(
    np.uint8
)


# Apply elevation-style colour map

rdsm_visual = cv2.applyColorMap(
    rdsm_8bit,
    cv2.COLORMAP_TURBO
)


# ============================================================
# 13. SAVE VISUAL rDSM
# ============================================================

rdsm_image_path = os.path.join(
    OUTPUT_FOLDER,
    "rDSM.png"
)

cv2.imwrite(
    rdsm_image_path,
    rdsm_visual
)


print(
    f"rDSM image saved to: {rdsm_image_path}"
)


# ============================================================
# 14. SAVE ORIGINAL IMAGE COPY
# ============================================================

original_path = os.path.join(
    OUTPUT_FOLDER,
    "original.png"
)

cv2.imwrite(
    original_path,
    original_image
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

cv2.imshow(
    "Original Image",
    original_image
)

cv2.imshow(
    "Relative DSM",
    rdsm_visual
)


print()
print("========================================")
print("rDSM GENERATION COMPLETE")
print("========================================")
print()
print("Output files:")
print("  output/original.png")
print("  output/depth_map.npy")
print("  output/rDSM.npy")
print("  output/rDSM.png")
print()
print("Press Q to close.")


# ============================================================
# 16. WAIT
# ============================================================

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# 17. CLEANUP
# ============================================================

cv2.destroyAllWindows()

print()
print("Program finished.")
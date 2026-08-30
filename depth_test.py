from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


print("Loading depth estimation model...")

# Load the pretrained depth estimation model
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("Model loaded!")


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image_path = "test.jpg"

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("Could not open the image.")
    print("Make sure test.jpg is in the same folder as this program.")
    print("Error:", e)
    exit()


print("Image loaded successfully!")
print("Image size:", image.size)


# --------------------------------------------------
# DEPTH ESTIMATION
# --------------------------------------------------

print("Estimating depth...")

result = depth_estimator(image)

# Extract depth image
depth_image = result["depth"]

# Convert PIL depth image to NumPy
depth = np.array(depth_image)

print("Depth estimation complete!")
print("Depth map shape:", depth.shape)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

plt.figure(figsize=(12, 5))


# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")


# Depth map
plt.subplot(1, 2, 2)
plt.imshow(depth, cmap="plasma")
plt.title("Estimated Depth")
plt.axis("off")


plt.tight_layout()
plt.show()


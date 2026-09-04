
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


# ============================================================
# 1. CHECK DEVICE
# ============================================================

if torch.cuda.is_available():
    device = 0
    print("Using NVIDIA GPU:", torch.cuda.get_device_name(0))
else:
    device = -1
    print("Using CPU")


# ============================================================
# 2. LOAD DEPTH MODEL
# ============================================================

print("Loading Depth Anything V2...")

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=device
)

print("Model loaded!")


# ============================================================
# 3. OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access camera.")
    exit()


# Lower camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started.")
print("Press Q to quit.")


# ============================================================
# 4. VARIABLES
# ============================================================

frame_counter = 0

depth_colored = None


# ============================================================
# 5. CAMERA LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read frame.")
        break


    frame_counter += 1


    # --------------------------------------------------------
    # Process every 5th frame
    # --------------------------------------------------------

    if frame_counter % 5 == 0:

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy → PIL
        image = Image.fromarray(frame_rgb)


        # ----------------------------------------------------
        # Resize image BEFORE sending to AI
        # ----------------------------------------------------

        small_image = image.resize((320, 240))


        # ----------------------------------------------------
        # Depth estimation
        # ----------------------------------------------------

        result = depth_estimator(small_image)

        depth_image = result["depth"]


        # ----------------------------------------------------
        # Convert depth to NumPy
        # ----------------------------------------------------

        depth = np.array(depth_image)


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        depth_normalized = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth_normalized = depth_normalized.astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Colorize
        # ----------------------------------------------------

        depth_colored = cv2.applyColorMap(
            depth_normalized,
            cv2.COLORMAP_PLASMA
        )


        # ----------------------------------------------------
        # Resize depth to camera display size
        # ----------------------------------------------------

        depth_colored = cv2.resize(
            depth_colored,
            (640, 480)
        )


    # ========================================================
    # 6. DISPLAY
    # ========================================================

    if depth_colored is not None:

        combined = np.hstack(
            (frame, depth_colored)
        )

        cv2.imshow(
            "Depth Wizard",
            combined
        )

    else:

        cv2.imshow(
            "Depth Wizard",
            frame
        )


    # ========================================================
    # 7. QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 8. CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("Depth Wizard stopped.")


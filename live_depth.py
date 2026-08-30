import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


# ============================================================
# 1. LOAD THE DEPTH MODEL
# ============================================================

print("Loading Depth Anything V2...")

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("Model loaded successfully!")


# ============================================================
# 2. OPEN THE WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the camera.")
    exit()

print("Camera started!")
print("Press Q to quit.")


# ============================================================
# 3. MAIN CAMERA LOOP
# ============================================================

while True:

    # Read one frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break


    # --------------------------------------------------------
    # Convert OpenCV BGR → RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # --------------------------------------------------------
    # Convert NumPy array → PIL Image
    # --------------------------------------------------------

    image = Image.fromarray(frame_rgb)


    # --------------------------------------------------------
    # RUN DEPTH AI
    # --------------------------------------------------------

    result = depth_estimator(image)

    depth_image = result["depth"]


    # --------------------------------------------------------
    # Convert depth map to NumPy
    # --------------------------------------------------------

    depth = np.array(depth_image)


    # --------------------------------------------------------
    # Normalize depth values for visualization
    # --------------------------------------------------------

    depth_normalized = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_normalized = depth_normalized.astype(np.uint8)


    # --------------------------------------------------------
    # Apply a color map
    # --------------------------------------------------------

    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_PLASMA
    )


    # --------------------------------------------------------
    # Resize depth map to camera size
    # --------------------------------------------------------

    depth_colored = cv2.resize(
        depth_colored,
        (frame.shape[1], frame.shape[0])
    )


    # ========================================================
    # 4. DISPLAY CAMERA + DEPTH
    # ========================================================

    combined = np.hstack((frame, depth_colored))


    cv2.imshow(
        "Depth Wizard - Live Depth",
        combined
    )


    # ========================================================
    # 5. QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# 6. CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("Depth Wizard stopped.") 

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import pipeline


# ============================================================
# 1. DEVICE
# ============================================================

device = 0 if torch.cuda.is_available() else -1

if device == -1:
    print("Using CPU")
else:
    print("Using CUDA")


# ============================================================
# 2. LOAD METRIC DEPTH MODEL
# ============================================================

print("Loading Depth Anything V2 Metric Outdoor Small...")

depth_model = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    device=device
)

print("Model loaded!")


# ============================================================
# 3. CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access camera.")
    exit()


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# 4. CAMERA INTRINSICS
# ============================================================

WIDTH = 640
HEIGHT = 480

fx = WIDTH
fy = WIDTH

cx = WIDTH / 2
cy = HEIGHT / 2


# ============================================================
# 5. FUNCTION:
#    DEPTH → 3D POINTS
# ============================================================

def depth_to_points(depth):

    h, w = depth.shape

    # Pixel coordinates

    u, v = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    # Convert to camera coordinates

    x = (u - cx) * depth / fx

    y = (v - cy) * depth / fy

    z = depth

    points = np.stack(
        (x, y, z),
        axis=-1
    )

    return points


# ============================================================
# 6. FUNCTION:
#    FIT GROUND PLANE
# ============================================================

def estimate_ground_height(points, depth):

    h, w = depth.shape

    # --------------------------------------------------------
    # Use lower portion of image.
    #
    # In a downward-looking camera, the ground is generally
    # concentrated toward the lower part of the image.
    # --------------------------------------------------------

    y_start = int(h * 0.55)

    ground_points = points[y_start:h]

    ground_points = ground_points.reshape(
        -1,
        3
    )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    valid = np.isfinite(
        ground_points
    ).all(axis=1)

    ground_points = ground_points[valid]


    if len(ground_points) < 100:

        return None


    # --------------------------------------------------------
    # Remove extreme depth values
    # --------------------------------------------------------

    z_values = ground_points[:, 2]

    median_z = np.median(z_values)

    valid_depth = (
        np.abs(z_values - median_z)
        < median_z * 0.5
    )

    ground_points = ground_points[
        valid_depth
    ]


    if len(ground_points) < 100:

        return None


    # --------------------------------------------------------
    # Fit plane using least squares
    #
    # Plane:
    #
    # Ax + By + Cz + D = 0
    # --------------------------------------------------------

    A = np.column_stack(
        (
            ground_points[:, 0],
            ground_points[:, 1],
            np.ones(len(ground_points))
        )
    )

    b = -ground_points[:, 2]


    try:

        coefficients, _, _, _ = np.linalg.lstsq(
            A,
            b,
            rcond=None
        )

    except np.linalg.LinAlgError:

        return None


    a = coefficients[0]
    b_plane = coefficients[1]
    c = 1.0


    # Plane:

    # ax + by + z + d = 0

    d = coefficients[2]


    # --------------------------------------------------------
    # Distance from camera origin (0,0,0)
    # to plane
    # --------------------------------------------------------

    numerator = abs(d)

    denominator = np.sqrt(
        a * a
        + b_plane * b_plane
        + c * c
    )

    height = numerator / denominator


    return height


# ============================================================
# 7. MAIN LOOP
# ============================================================

print()
print("Camera started!")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:

        print("Could not read frame.")

        break


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT)
    )


    # --------------------------------------------------------
    # Convert OpenCV → PIL
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)


    # --------------------------------------------------------
    # Depth estimation
    # --------------------------------------------------------

    result = depth_model(image)


    # --------------------------------------------------------
    # Get depth
    # --------------------------------------------------------

    depth = result["predicted_depth"]

    depth = depth.squeeze().cpu().numpy()


    # --------------------------------------------------------
    # Resize depth
    # --------------------------------------------------------

    depth = cv2.resize(
        depth,
        (WIDTH, HEIGHT)
    )


    # --------------------------------------------------------
    # Convert depth to 3D
    # --------------------------------------------------------

    points = depth_to_points(
        depth
    )


    # --------------------------------------------------------
    # Estimate camera height
    # --------------------------------------------------------

    estimated_height = estimate_ground_height(
        points,
        depth
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    if estimated_height is not None:

        text = (
            f"Estimated Height: "
            f"{estimated_height:.2f} m"
        )

        cv2.putText(
            frame,
            text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Ground not detected",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # Show camera
    # --------------------------------------------------------

    cv2.imshow(
        "Depth Wizard - Height Estimation",
        frame
    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 8. CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("Height estimation stopped.")
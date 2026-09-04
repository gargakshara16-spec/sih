
import cv2
import torch
import numpy as np


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cpu")

print("Using CPU")
print("Loading MiDaS Small...")


# ============================================================
# 2. LOAD MiDaS
# ============================================================

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.to(device)
midas.eval()


# ============================================================
# 3. LOAD TRANSFORMATION
# ============================================================

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform

print("MiDaS loaded successfully!")


# ============================================================
# 4. CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access camera.")
    exit()


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


print("Camera started!")
print("Move your cursor over the camera.")
print("Press Q to quit.")


# ============================================================
# 5. MOUSE VARIABLES
# ============================================================

mouse_x = 320
mouse_y = 240

current_depth = 0
current_depth_percent = 0


# ============================================================
# 6. MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global mouse_x, mouse_y

    if event == cv2.EVENT_MOUSEMOVE:

        mouse_x = x
        mouse_y = y


# Create the camera window
cv2.namedWindow("Depth Wizard")

# Connect mouse movement to our function
cv2.setMouseCallback(
    "Depth Wizard",
    mouse_callback
)


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read frame.")
        break


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    img = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Prepare image for MiDaS
    # --------------------------------------------------------

    input_batch = transform(img).to(device)


    # --------------------------------------------------------
    # Depth inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()


    # --------------------------------------------------------
    # Convert prediction to NumPy
    # --------------------------------------------------------

    depth = prediction.cpu().numpy()


    # ========================================================
    # 8. NORMALIZE DEPTH
    # ========================================================

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


    # ========================================================
    # 9. COLOR DEPTH MAP
    # ========================================================

    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_PLASMA
    )


    # ========================================================
    # 10. GET DEPTH AT CURSOR POSITION
    # ========================================================

    # Make sure cursor remains inside image
    x = max(0, min(mouse_x, frame.shape[1] - 1))
    y = max(0, min(mouse_y, frame.shape[0] - 1))


    # Get the depth value at the selected pixel
    current_depth = int(
        depth_normalized[y, x]
    )


    # Convert to a percentage
    current_depth_percent = (
        current_depth / 255
    ) * 100


    # ========================================================
    # 11. DRAW CROSSHAIR
    # ========================================================

    cv2.drawMarker(
        frame,
        (x, y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=15,
        thickness=2
    )


    # ========================================================
    # 12. DISPLAY DEPTH NEXT TO CURSOR
    # ========================================================

    cursor_text = (
        f"Depth: {current_depth_percent:.1f}%"
    )


    cv2.putText(
        frame,
        cursor_text,
        (x + 15, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 13. BOTTOM-RIGHT INFORMATION BOX
    # ========================================================

    box_width = 250
    box_height = 75

    frame_height, frame_width = frame.shape[:2]

    box_x1 = frame_width - box_width - 10
    box_y1 = frame_height - box_height - 10

    box_x2 = frame_width - 10
    box_y2 = frame_height - 10


    # Semi-transparent black box
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (0, 0, 0),
        -1
    )

    # Blend box with camera
    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )


    # ========================================================
    # 14. TEXT INSIDE BOTTOM-RIGHT BOX
    # ========================================================

    cv2.putText(
        frame,
        "DEPTH WIZARD",
        (box_x1 + 10, box_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Relative Depth: {current_depth_percent:.1f}%",
        (box_x1 + 10, box_y1 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # 15. DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Depth Wizard",
        frame
    )


    # ========================================================
    # 16. QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 17. CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("Depth Wizard stopped.")

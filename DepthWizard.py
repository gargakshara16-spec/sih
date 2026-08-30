message = "Welcome to Python!"
print(message)

import cv2

# Open the default camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access the camera.")
    exit()

print("Camera started!")
print("Press Q to quit.")

while True:
    # Read a frame
    ret, frame = camera.read()

    if not ret:
        print("Could not read frame.")
        break

    # Display the camera feed
    cv2.imshow("My Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()
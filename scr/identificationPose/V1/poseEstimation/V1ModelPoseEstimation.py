from ultralytics import YOLO
import cv2
import os

# Path to the model
modelPath = "../../yolo11m-pose.pt"
modelAbsolutePath = os.path.abspath(modelPath)

model = YOLO(modelAbsolutePath)

#Aceess the camera
camera = cv2.VideoCapture(0)

# Verify if the camera is opened
if not camera.isOpened():
    print("Could not open camera")
    exit()

print("Press 'q' to quit")

# Read de frames from the camera
while True:
    ret, frame = camera.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect the pose in the frame
    results = model.predict(source= frame, save=False, conf=0.5)

    # Plot the pose in the frame
    annotatedImage = results[0].plot()

    cv2.imshow("Annotated Image", annotatedImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()
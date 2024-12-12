from ultralytics import YOLO
import os

modelPath = "../../yolo11m-pose.pt"

modelAbsolutePath = os.path.abspath(modelPath)

# Load a model
model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO(modelAbsolutePath)  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-pose.yaml").load(modelAbsolutePath)  # build from YAML and transfer weights

# Train the model
results = model.train(data="hand-keypoints.yaml", epochs=10, imgsz=640)


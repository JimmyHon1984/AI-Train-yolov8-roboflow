from ultralytics import YOLO

# Load the YOLOv5 model
model = YOLO("yolov8s.pt")

# Export the model to TF.js format
model.export(format="tfjs")
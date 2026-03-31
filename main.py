import os
from ultralytics import YOLO

path = "./Predict"
model = YOLO('YOLO_Custom_v8m.pt')

for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing: {filename}")
        model.predict(source=filepath, save=True, conf=0.37)

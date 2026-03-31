from ultralytics import YOLO
import locale

locale.getpreferredencoding = lambda: "UTF-8"

# List of YOLO models
models = [
    "yolov8n.pt",  # nano
    "yolov8s.pt",  # small
    "yolov8m.pt",  # medium
    "yolov8l.pt",  # large
    "yolov8x.pt"   # extra large
]

# OPTIONAL (only if your teacher forces "YOLOv11")
# models.append("yolo11n.pt")  # may NOT work unless available

for model_name in models:
    print(f"\n🚀 Training {model_name}...\n")
    
    model = YOLO(model_name)
    
    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=30,       # keep low for time
        patience=10,
        name=model_name.split(".")[0]  # separate folders
    )

print("\n✅ All models trained successfully!")
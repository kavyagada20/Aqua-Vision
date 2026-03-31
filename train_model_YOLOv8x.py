from ultralytics import YOLO
import locale
locale.getpreferredencoding = lambda: "UTF-8"

model = YOLO('yolov8x.pt')

model.train(
    data='data.yaml',
    imgsz=640,
    epochs=50,
    patience=20
)
from roboflow import Roboflow
from ultralytics import YOLO
from apikey import ROBOFLOW_API_KEY, get
model = YOLO(model='yolo11n.pt',task='detect')
rf_key = get("ROBOFLOW_API_KEY")
rf = Roboflow(api_key = rf_key)
project = rf.workspace("samsung-competition").project("my-first-project-9zbk7")
version = project.version(2)
dataset = version.download("yolov11")

model.train(
    data='data.yaml',
    device=0,
    epochs=150,
    amp=True,          # ✅ FP16로 VRAM 절약
    deterministic=True,
    batch=4,           # 그대로 두고
    imgsz=512,         # 그대로 (여전히 부족하면 448/416)
    workers=0
)


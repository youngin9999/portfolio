from roboflow import Roboflow
from ultralytics import YOLO

model = YOLO(model='yolo11n.pt',task='detect')
rf = Roboflow(api_key="LTYU5Wr6vTjeqXzk6n8G")
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


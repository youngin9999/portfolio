import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import pytesseract
from pdf2image import convert_from_path
import subprocess
from pathlib import Path

# 모델 로딩
model = YOLO("./model/yolov11n-doclaynet.pt")

# 클래스 매핑
LABEL_MAP = {
    'Text': 'text',
    'Title': 'title',
    'Section-header': 'subtitle',
    'Formula': 'equation',
    'Table': 'table',
    'Picture': 'image'
}

def convert_to_images(input_path, temp_dir, dpi=200):
    ext = Path(input_path).suffix.lower()
    os.makedirs(temp_dir, exist_ok=True)

    if ext == ".pdf":
        return convert_from_path(input_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext == ".pptx":
        # Convert pptx to pdf first
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path
        ], check=True)
        pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
        return convert_from_path(pdf_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext in [".jpg", ".jpeg", ".png"]:
        return [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

def scale_bbox_to_target(bbox, current_size, target_size):
    x1, y1, x2, y2 = bbox
    scale_x = target_size[0] / current_size[0]
    scale_y = target_size[1] / current_size[1]
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]

def extract_text(image_pil, bbox):
    x1, y1, x2, y2 = bbox
    cropped = image_pil.crop((x1, y1, x2, y2))
    return pytesseract.image_to_string(cropped, lang='kor+eng').strip()

def inference_one_image(id_val, image_pil, target_size, conf_thres=0.3):
    original_size = image_pil.size  # (w, h)
    resized_image = image_pil.resize((1024, 1024))
    temp_path = "_temp_image.png"
    resized_image.save(temp_path)

    results = model(source=temp_path, imgsz=1024, conf=conf_thres, verbose=False)[0]
    os.remove(temp_path)

    predictions = []
    order = 0
    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        label = results.names[int(cls)]
        if label not in LABEL_MAP:
            continue
        category_type = LABEL_MAP[label]
        x1, y1, x2, y2 = scale_bbox_to_target(box.tolist(), (1024, 1024), target_size)
        text = extract_text(image_pil, [x1, y1, x2, y2]) if category_type in ['title', 'subtitle', 'text'] else ''
        predictions.append({
            'ID': id_val,
            'category_type': category_type,
            'confidence_score': score.cpu().item(),
            'order': order,
            'text': text,
            'bbox': f'{x1}, {y1}, {x2}, {y2}'
        })
        order += 1
    return predictions

def inference(test_csv_path="./data/test.csv", output_csv_path="./output/submission.csv"):
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_image_dir = "./temp_images"
    os.makedirs(temp_image_dir, exist_ok=True)

    csv_dir = os.path.dirname(test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    all_preds = []

    for _, row in test_df.iterrows():
        id_val = row['ID']
        raw_path = row['path']
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))
        target_width = int(row['width'])
        target_height = int(row['height'])

        if not os.path.exists(file_path):
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        try:
            images = convert_to_images(file_path, temp_image_dir)
            for i, image in enumerate(images):
                full_id = f"{id_val}_p{i+1}" if len(images) > 1 else id_val
                preds = inference_one_image(full_id, image, (target_width, target_height))
                all_preds.extend(preds)
            print(f"✅ 예측 완료: {file_path}")
        except Exception as e:
            print(f"❌ 처리 실패: {file_path} → {e}")

    result_df = pd.DataFrame(all_preds)
    result_df.to_csv(output_csv_path, index=False, encoding='UTF-8-sig')
    print(f"✅ 저장 완료: {output_csv_path}")

if __name__ == "__main__":
    inference("./data/test.csv", "./output/submission.csv")
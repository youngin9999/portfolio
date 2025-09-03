
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import subprocess
import easyocr

# ==============================
# Paths & Constants
# ==============================
VIZ_DIR = "/workspace/util/viz"
KOR_FONT_PATH = "/workspace/util/viz/NanumGothic.otf"  # Korean font path provided by user
TEST_CSV_PATH = "./data/test.csv"
OUTPUT_CSV_PATH = "./output/submission.csv"
TEMP_IMAGE_DIR = "./temp_images"

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# (Optional / Example) Tesseract config kept for reference; not used in this script.
# tessdata_dir_config = r'--oem 1 --tessdata-dir "/workspace/util/model/ocr"'
# img = cv2.imread('/workspace/util/data/test/TEST_03.jpg')
# if img is not None:
#     dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==============================
# Utilities
# ==============================
def convert_to_images(input_path: str, temp_dir: str, dpi: int = 400):
    """
    Convert supported inputs to a list of PIL Images.
    - PDF: uses poppler (pdf2image)
    - PPTX: uses LibreOffice to convert to PDF, then pdf2image
    - JPG/PNG: load directly
    """
    ext = Path(input_path).suffix.lower()
    os.makedirs(temp_dir, exist_ok=True)

    if ext == ".pdf":
        return convert_from_path(input_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext == ".pptx":
        # Convert pptx to pdf first
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path],
            check=True
        )
        pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
        return convert_from_path(pdf_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext in [".jpg", ".jpeg", ".png"]:
        return [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def draw_korean_text_on_image(vis_rgb: np.ndarray, results, font_path: str) -> np.ndarray:
    """
    Draw OCR text labels (Korean/Unicode) on an RGB image using PIL (for proper font rendering).
    - vis_rgb: RGB numpy array
    - results: EasyOCR readtext results [(bbox, text, conf), ...]
    - font_path: path to .ttf/.otf font that supports Korean
    Returns: RGB numpy array with text rendered.
    """
    pil_img = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(pil_img)
    H, W = vis_rgb.shape[:2]
    font_size = max(16, int(min(H, W) * 0.028))
    font = ImageFont.truetype(font_path, font_size)

    for (bbox, text, conf) in results:
        tl = tuple(map(int, bbox[0]))
        draw.text(
            (tl[0], max(0, tl[1] - 10)),
            text,
            font=font,
            fill=(255, 0, 0),           # red
            stroke_width=1,
            stroke_fill=(255, 255, 255) # white outline for readability
        )

    return np.array(pil_img)

# ==============================
# Main
# ==============================
def main():
    # Load test CSV
    csv_dir = os.path.dirname(TEST_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Initialize EasyOCR (GPU if available; fallback to CPU for robustness)
    try:
        reader = easyocr.Reader(['ko', 'en'], gpu=True, model_storage_directory="/workspace/util/model/ocr")
    except Exception:
        reader = easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory="/workspace/util/model/ocr")

    all_preds = []

    for _, row in test_df.iterrows():
        id_val = row['ID']
        raw_path = row['path']
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))

        # These are currently not used; keep if needed later for resizing/normalization
        # target_width = int(row['width'])
        # target_height = int(row['height'])

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        try:
            images = convert_to_images(file_path, TEMP_IMAGE_DIR)

            for i, image in enumerate(images):
                # Ensure RGB numpy array
                image_np = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image
                result = reader.readtext(image_np)

                # 1) Draw rectangles using OpenCV
                vis = image_np.copy()
                for (bbox, text, conf) in result:
                    tl = tuple(map(int, bbox[0]))
                    br = tuple(map(int, bbox[2]))
                    cv2.rectangle(vis, tl, br, (0, 255, 0), 2)

                    # accumulate results for CSV
                    all_preds.append({
                        "ID": id_val,
                        "page": i,
                        "text": text,
                        "conf": float(conf),
                        "x1": tl[0], "y1": tl[1], "x2": br[0], "y2": br[1],
                    })

                # 2) Draw Korean text using PIL font
                vis = draw_korean_text_on_image(vis, result, KOR_FONT_PATH)

                # 3) Save visualization
                out_path = os.path.join(VIZ_DIR, f"{Path(file_path).stem}_{i}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"❌ Failed: {file_path} → {e}")

    # Save CSV
    result_df = pd.DataFrame(all_preds)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="UTF-8-sig")
    print(f"✅ Saved: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()

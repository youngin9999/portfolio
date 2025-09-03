
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

def render_text_only_canvas(results, canvas_size, font_path: str, bg=(255, 255, 255)) -> np.ndarray:
    """
    Create a blank canvas and render ONLY the recognized text at its positions.
    - results: EasyOCR readtext results [(bbox, text, conf), ...]
    - canvas_size: (H, W) tuple (same size as original image)
    - font_path: Korean-capable font
    - bg: background color
    Returns: RGB numpy array (H, W, 3)
    """
    H, W = canvas_size
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    font_size = max(16, int(min(H, W) * 0.028))
    font = ImageFont.truetype(font_path, font_size)

    for (bbox, text) in results:
        tl = tuple(map(int, bbox[0]))
        draw.text(
            (tl[0], max(0, tl[1] - 10)),
            text,
            font=font,
            fill=(0, 0, 0),  # black text
            stroke_width=0
        )
    return np.array(pil_img)

def side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray, sep: int = 6) -> np.ndarray:
    """
    Concatenate left and right RGB images with a thin separator.
    Assumes both images have the same height/width.
    """
    H, W = left_rgb.shape[:2]
    separator = np.full((H, sep, 3), 220, dtype=np.uint8)  # light gray separator
    combined = np.hstack([left_rgb, separator, right_rgb])
    return combined

# ==============================
# Main
# ==============================
def main():
    # Load test CSV
    csv_dir = os.path.dirname(TEST_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Initialize EasyOCR (GPU if available; fallback to CPU)
    try:
        reader = easyocr.Reader(['ko', 'en'], gpu=True, model_storage_directory="/workspace/util/model/ocr" )
    except :
        reader = easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory="/workspace/util/model/ocr" )
    all_preds = []

    for _, row in test_df.iterrows():
        id_val = row['ID']
        raw_path = row['path']
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        try:
            images = convert_to_images(file_path, TEMP_IMAGE_DIR)

            for i, image in enumerate(images):
                # Ensure RGB numpy array
                image_np = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image
                result = reader.readtext(image_np , paragraph= True , )

                # LEFT: original + boxes only (NO text)
                left = image_np.copy()
                for (bbox, text) in result:
                    tl = tuple(map(int, bbox[0]))
                    br = tuple(map(int, bbox[2]))
                    cv2.rectangle(left, tl, br, (0, 255, 0), 2)

                    # Accumulate CSV rows
                    all_preds.append({
                        "ID": id_val,
                        "page": i,
                        "text": text,
                        "conf": float(50),
                        "x1": tl[0], "y1": tl[1], "x2": br[0], "y2": br[1],
                    })

                # RIGHT: text-only on blank canvas
                right = render_text_only_canvas(result, image_np.shape[:2], KOR_FONT_PATH, bg=(255, 255, 255))

                # Compose and save
                sbs = side_by_side(left, right, sep=6)
                out_path = os.path.join(VIZ_DIR, f"{Path(file_path).stem}_{i}_sbs_leftBoxes_rightText.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"❌ Failed: {file_path} → {e}")

    # Save CSV
    result_df = pd.DataFrame(all_preds)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="UTF-8-sig")
    print(f"✅ Saved: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()

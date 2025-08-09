#!/usr/bin/env python3
"""
Prototype (pip-only, sample path semantics, PaddleOCR v2/v3 compatible)
- YOLO (Ultralytics) for 6 classes: title, subtitle, text, image, table, equation
- OCR: PaddleOCR (v2 or v3 both OK)
- PDF render: PyMuPDF (pymupdf)
- PPTX: not rendered (recommend pre-convert to PDF/PNG and update test.csv path)
- Reads files like the official sample: file_path = dirname(test.csv) / path
- Outputs: ID, category_type, confidence_score, order, text, bbox ("x1, y1, x2, y2")

Run example (from project root):
  python yolo.py --weights ./model/yolov11n-doclaynet.pt --data_csv ./data/test.csv --out_csv ./output/submission.csv --device 0

Dependencies (pip-only):
  pip install ultralytics paddleocr pymupdf opencv-python-headless numpy pandas tqdm
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO

# PyMuPDF import (package name is 'pymupdf', import name is 'fitz')
try:
    import pymupdf as fitz
except Exception:
    import fitz  # type: ignore

# PaddleOCR (v2/v3 compatibility)
from paddleocr import PaddleOCR
try:
    from paddleocr import __version__ as paddleocr_version
except Exception:
    paddleocr_version = "2.0.0"

# =====================
# Config / Constants
# =====================
CATEGORY_NAMES = ["title", "subtitle", "text", "image", "table", "equation"]
TEXT_CATS = {"title", "subtitle", "text"}

# Map common dataset labels (e.g., DocLayNet) to our 6-class schema
NAME_CANON = {
    "Text": "text",
    "Title": "title",
    "Section-header": "subtitle",
    "Formula": "equation",
    "Table": "table",
    "Picture": "image",
}


@dataclass
class Det:
    cls_name: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    text: str = ""
    order: int = -1

    def as_row(self, page_id: str) -> Tuple[str, str, float, int, str, str]:
        bbox_str = f"{self.x1}, {self.y1}, {self.x2}, {self.y2}"
        return (
            page_id,
            self.cls_name,
            round(self.conf, 6),
            int(self.order if self.order >= 0 else 0),
            self.text,
            bbox_str,
        )


# =====================
# Utilities
# =====================

def normalize_device(dev):
    s = str(dev).lower()
    if s in {"cpu", "-1"}:
        return "cpu"
    if s.startswith("cuda"):
        return s
    if s.isdigit():
        return int(s)
    return "cpu"


def load_yolo(weights: Path):
    """Return YOLO model; device is passed at predict-time (no .to())."""
    return YOLO(str(weights))


def init_ocr(lang: str = "korean") -> PaddleOCR:
    """Create PaddleOCR instance with v2/v3 compatibility."""
    # Parse major.minor
    try:
        parts = [int(x) for x in paddleocr_version.split(".")[:2]]
        ver_tuple = (parts[0], parts[1])
    except Exception:
        ver_tuple = (2, 0)

    if ver_tuple >= (3, 0):
        # v3 API: det/rec/use_angle_cls removed, use_textline_orientation added
        return PaddleOCR(use_textline_orientation=False, lang=lang)
    else:
        # v2 API: legacy flags
        return PaddleOCR(show_log=False, use_angle_cls=True, lang=lang, det=False, rec=True)


def pdf_to_images_with_fitz(pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def convert_to_images(input_path: Path, dpi: int = 200) -> List[Image.Image]:
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        return pdf_to_images_with_fitz(input_path, dpi=dpi)
    elif ext in {".jpg", ".jpeg", ".png"}:
        return [Image.open(str(input_path)).convert("RGB")]
    elif ext == ".pptx":
        raise RuntimeError(
            f"PPTX rendering requires LibreOffice (not pip-installable). "
            f"Pre-convert {input_path.name} to PDF/PNG and update test.csv path."
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def run_ocr_text(ocr: PaddleOCR, image_pil: Image.Image, bbox_xyxy: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    crop = np.array(image_pil.crop((x1, y1, x2, y2)))

    # Try v2-style first
    try:
        res = ocr.ocr(crop, det=False, rec=True)
        if res and res[0]:
            return " ".join([t for (t, c) in res[0]]).strip()
    except Exception:
        pass

    # Fallback to v3-style predict output parsing
    try:
        out = ocr.predict(crop)
        # Expect list-like with dict or .res containing 'rec_texts'
        texts = []
        if isinstance(out, list) and out:
            o0 = out[0]
            if hasattr(o0, "res") and isinstance(o0.res, dict):
                texts = o0.res.get("rec_texts", []) or []
            elif isinstance(o0, dict):
                texts = o0.get("rec_texts", []) or []
        elif isinstance(out, dict):
            texts = out.get("rec_texts", []) or []
        return " ".join([str(t) for t in texts]).strip()
    except Exception:
        return ""


def scale_bbox_to_target(bbox_xyxy: np.ndarray, curr_wh: Tuple[int, int], target_wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    cw, ch = curr_wh
    tw, th = target_wh
    sx = tw / float(cw)
    sy = th / float(ch)
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
    return X1, Y1, X2, Y2


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), W - 1))
    x2 = max(0, min(int(x2), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def assign_reading_order(dets: List[Det], page_w: int) -> None:
    if not dets:
        return
    centers = np.array([(d.x1 + d.x2) / 2 for d in dets], dtype=np.float32)
    xs = np.sort(centers)
    two_col = (len(xs) > 1 and np.max(np.diff(xs)) > 0.22 * page_w)

    if two_col:
        idx_sort = np.argsort(centers)
        gaps = np.diff(centers[idx_sort])
        split_pos = int(np.argmax(gaps)) + 1
        left_idx = idx_sort[:split_pos]
        right_idx = idx_sort[split_pos:]
        groups = [list(left_idx), list(right_idx)]
    else:
        groups = [list(range(len(dets)))]

    order = 0
    for g in groups:
        g_sorted = sorted(g, key=lambda i: (dets[i].y1, dets[i].x1))
        for i in g_sorted:
            dets[i].order = order
            order += 1


# =====================
# Inference core (per image)
# =====================

def detect_on_pil(model: YOLO, image_pil: Image.Image, imgsz: int, conf: float, iou: float,
                   max_det: int, device: str | int, target_wh: Tuple[int, int],
                   ocr: PaddleOCR) -> List[Det]:
    img_np = np.array(image_pil)  # RGB
    dev = normalize_device(device)
    res = model.predict(img_np, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                        device=dev, verbose=False)[0]

    H, W = res.orig_shape
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), int)
    confs = res.boxes.conf.cpu().numpy().astype(float) if res.boxes is not None else np.zeros((0,), float)

    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}

    out: List[Det] = []
    for box, cls, cf in zip(boxes, clss, confs):
        raw = str(names.get(int(cls), int(cls)))
        name = NAME_CANON.get(raw, raw.lower())
        if name not in CATEGORY_NAMES:
            continue
        x1, y1, x2, y2 = scale_bbox_to_target(box, (W, H), target_wh)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, *target_wh)
        det = Det(name, float(cf), x1, y1, x2, y2)
        if det.cls_name in TEXT_CATS:
            det.text = run_ocr_text(ocr, image_pil, (x1, y1, x2, y2))
        out.append(det)

    assign_reading_order(out, page_w=target_wh[0])
    return out


# =====================
# Main CLI
# =====================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data_csv", default=str(Path(__file__).resolve().parent / "data" / "test.csv"))
    p.add_argument("--out_csv", default=str(Path(__file__).resolve().parent / "output" / "submission.csv"))
    p.add_argument("--imgsz", type=int, default=896)
    p.add_argument("--device", default="cpu")  # default to CPU; pass 0 or cuda:0 to use GPU
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=3000)
    p.add_argument("--lang", default="korean")
    args = p.parse_args()

    model = load_yolo(Path(args.weights))
    ocr = init_ocr(args.lang)

    data_csv = Path(args.data_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    need_cols = {"ID", "path", "width", "height"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"{data_csv} must contain columns: {need_cols}")

    rows: List[Tuple[str, str, float, int, str, str]] = []
    csv_dir = data_csv.parent

    for _, r in tqdm(df.iterrows(), total=len(df), desc="infer"):
        page_id = str(r["ID"])  # keep ID as-is even for multi-page
        raw_path = str(r["path"])  # e.g., ./test/TEST_01.png
        file_path = (csv_dir / raw_path).resolve()
        target_wh = (int(r["width"]), int(r["height"]))

        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        try:
            images = convert_to_images(file_path)
        except Exception as e:
            print(f"[WARN] Convert failed: {file_path} -> {e}")
            continue




        for img in images:
            dets = detect_on_pil(model, img, args.imgsz, args.conf, args.iou,
                                 args.max_det, args.device, target_wh, ocr)
            dets.sort(key=lambda d: d.order if d.order >= 0 else 10**9)
            for d in dets:
                rows.append(d.as_row(page_id))

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "category_type", "confidence_score", "order", "text", "bbox"])
        w.writerows(rows)




    print(f"[OK] wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()

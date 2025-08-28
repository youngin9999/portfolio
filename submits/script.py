#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script.py  —  Document layout + OCR (Tesseract) → submission.csv
- EasyOCR 대신 Tesseract 사용
- PDF: poppler(pdf2image) → PyMuPDF 폴백
- PPTX: LibreOffice(soffice) → PDF → 이미지
- doclayout-yolo 있으면 사용, 없으면 ultralytics YOLO 폴백
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Model backend: doclayout-yolo → fallback to ultralytics
try:
    from doclayout_yolo import YOLOv10 as _YOLO
    _BACKEND = "doclayout-yolo"
except Exception:
    from ultralytics import YOLO as _YOLO
    _BACKEND = "ultralytics"

# OCR: Tesseract
import pytesseract

# PDF → image
from pdf2image import convert_from_path
try:
    import pymupdf as fitz  # PyMuPDF (new import name)
except Exception:
    try:
        import fitz  # type: ignore
    except Exception:
        fitz = None


# ===================== Settings / Mappings =====================
CATEGORY_NAMES = ["title", "subtitle", "text", "image", "table", "equation"]
TEXT_CATS = {"title", "subtitle", "text"}

# doclayout-yolo 원본 라벨을 우리 6클래스로
_RAW2CANON = {
    "title": "title",
    "plain text": "text",
    "paragraph": "text",
    "figure": "image",
    "figure caption": "text",
    "table": "table",
    "table caption": "text",
    "table footnote": "text",
    "isolated formula": "equation",
    "formula": "equation",
    "formula caption": "text",
    "abandoned text": None,  # 제출 제외
}
def _norm_key(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").split())
def remap_label(raw: str) -> str | None:
    k = _norm_key(raw)
    mapped = _RAW2CANON.get(k, None)
    if mapped is None:
        return None
    return mapped if mapped in CATEGORY_NAMES else None


# ===================== Data classes =====================
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
            round(float(self.conf), 6),
            int(self.order if self.order >= 0 else 0),
            self.text,
            bbox_str,
        )


# ===================== OCR (Tesseract) =====================
@dataclass
class TesseractCfg:
    lang: str = "eng"   # e.g., "kor+eng"
    oem: int = 3        # 3: LSTM
    psm: int = 6        # 6: uniform block
    extra: str = ""     # 추가 옵션
    tesseract_cmd: str | None = None

def _langs_to_tesseract_code(langs: str) -> str:
    m = {
        "ko": "kor", "kor": "kor", "korean": "kor",
        "en": "eng", "eng": "eng", "english": "eng",
        "ja": "jpn", "zh": "chi_sim",
    }
    parts = [p.strip().lower() for p in (langs or "eng").split(",") if p.strip()]
    mapped = [m.get(p, p) for p in parts]
    return "+".join(mapped) if mapped else "kor+eng"

def init_ocr(langs="ko,en", tesseract_cmd="", oem=3, psm=6, extra="") -> TesseractCfg:
    cfg = TesseractCfg(lang=_langs_to_tesseract_code(langs), oem=int(oem), psm=int(psm), extra=str(extra))
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        cfg.tesseract_cmd = tesseract_cmd
    return cfg

def run_ocr_text(reader: TesseractCfg, image_pil: Image.Image, bbox_xyxy: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    crop = image_pil.crop((x1, y1, x2, y2))
    config = f"--oem {reader.oem} --psm {reader.psm}"
    if reader.extra:
        config = f"{config} {reader.extra}"
    try:
        text = pytesseract.image_to_string(crop, lang=reader.lang, config=config)
        text = " ".join(t.strip() for t in text.splitlines() if t.strip())
        return text
    except Exception:
        return ""


# ===================== Utilities =====================
BASE_DIR = Path(__file__).resolve().parent
##
def normalize_device(dev):
    s = str(dev).lower()
    if s in {"cpu", "-1"}: return "cpu"
    if s.startswith("cuda"): return s
    if s.isdigit(): return int(s)
    return "cpu"

def load_model(weights: Path):
    model = _YOLO(str(weights))
    print(f"[INFO] backend={_BACKEND}, weights={weights}")
    return model

def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None

def pdf_to_images(pdf_path: Path, dpi: int, poppler_path: str | None) -> List[Image.Image]:
    # pdf2image (Poppler) 1차 시도
    try:
        kw = {"dpi": dpi}
        if poppler_path:
            kw["poppler_path"] = poppler_path
        return convert_from_path(str(pdf_path), **kw)
    except Exception as e:
        # PyMuPDF 폴백
        if fitz is None:
            raise RuntimeError(f"PDF convert failed (no PyMuPDF fallback): {e}")
        doc = fitz.open(str(pdf_path))
        pages = []
        for p in doc:
            pix = p.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        return pages

def convert_to_images(
    input_path: Path,
    dpi: int = 200,
    poppler_path: str | None = None,
    soffice_path: str | None = None,
) -> List[Image.Image]:
    ext = input_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return [Image.open(str(input_path)).convert("RGB")]
    if ext == ".pdf":
        return pdf_to_images(input_path, dpi=dpi, poppler_path=poppler_path)
    if ext == ".pptx":
        exe = soffice_path or shutil.which("soffice") or shutil.which("libreoffice")
        if not exe:
            raise RuntimeError("LibreOffice(soffice) not found in PATH. Install or pass --soffice_path.")
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            cmd = [
                exe, "--headless", "--invisible", "--norestore",
                "--nolockcheck", "--nodefault", "--nofirststartwizard",
                "--convert-to", "pdf", "--outdir", str(outdir), str(input_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pdf_path = outdir / (input_path.stem + ".pdf")
            if not pdf_path.exists():
                raise RuntimeError(f"LibreOffice failed to export PDF: {input_path.name}")
            return pdf_to_images(pdf_path, dpi=dpi, poppler_path=poppler_path)
    raise ValueError(f"Unsupported file type: {ext}")

def scale_bbox_to_target(bbox_xyxy: np.ndarray, curr_wh: Tuple[int, int], target_wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    cw, ch = curr_wh
    tw, th = target_wh
    sx = tw / float(cw); sy = th / float(ch)
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
    return X1, Y1, X2, Y2

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), W - 1)); x2 = max(0, min(int(x2), W - 1))
    y1 = max(0, min(int(y1), H - 1)); y2 = max(0, min(int(y2), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
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
        left_idx = idx_sort[:split_pos]; right_idx = idx_sort[split_pos:]
        groups = [list(left_idx), list(right_idx)]
    else:
        groups = [list(range(len(dets)))]
    order = 0
    for g in groups:
        g_sorted = sorted(g, key=lambda i: (dets[i].y1, dets[i].x1))
        for i in g_sorted:
            dets[i].order = order; order += 1


# ===================== Detection =====================
def detect_on_pil(
    model: Any,
    image_pil: Image.Image,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str | int,
    target_wh: Tuple[int, int],
    ocr_engine: TesseractCfg,
) -> List[Det]:
    img_np = np.array(image_pil)  # RGB
    dev = normalize_device(device)
    res = model.predict(img_np, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=dev, verbose=False)[0]

    H, W = res.orig_shape
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    clss  = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), int)
    confs = res.boxes.conf.cpu().numpy().astype(float) if res.boxes is not None else np.zeros((0,), float)
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}

    out: List[Det] = []
    for box, cls, cf in zip(boxes, clss, confs):
        raw = str(names.get(int(cls), int(cls)))
        name = remap_label(raw)
        if name is None:
            continue
        x1, y1, x2, y2 = scale_bbox_to_target(box, (W, H), target_wh)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, *target_wh)
        det = Det(name, float(cf), x1, y1, x2, y2)
        if det.cls_name in TEXT_CATS:
            det.text = run_ocr_text(ocr_engine, image_pil, (x1, y1, x2, y2))
        out.append(det)

    assign_reading_order(out, page_w=target_wh[0])
    return out


# ===================== Main =====================
def main():
    default_weights = BASE_DIR / "model" / "yolo" / "doclayout_yolo_docstructbench_imgsz1024.pt"
    default_data = BASE_DIR / "data" / "test.csv"
    default_out  = BASE_DIR / "output" / "submission.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(default_weights))
    ap.add_argument("--data_csv", default=str(default_data))
    ap.add_argument("--out_csv",  default=str(default_out))
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--device", default="cpu")  # 안전 기본값
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.6)
    ap.add_argument("--max_det", type=int, default=3000)

    # 렌더 옵션
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--poppler_path", default="")
    ap.add_argument("--soffice_path", default="")

    # OCR (Tesseract)
    ap.add_argument("--ocr_langs", default="ko,en")
    ap.add_argument("--tesseract_cmd", default="")  # 비표준 경로면 지정
    ap.add_argument("--oem", type=int, default=3)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--ocr_extra", default="")

    # 자동 시각화
    ap.add_argument("--auto_viz", action="store_true")
    args = ap.parse_args()

    # 준비
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 모델 / OCR
    model = load_model(Path(args.weights))
    ocr = init_ocr(langs=args.ocr_langs, tesseract_cmd=args.tesseract_cmd, oem=args.oem, psm=args.psm, extra=args.ocr_extra)

    # 데이터 읽기
    data_csv = Path(args.data_csv)
    df = pd.read_csv(data_csv)
    need = {"ID", "path", "width", "height"}
    if not need.issubset(df.columns):
        raise ValueError(f"{data_csv} must contain columns: {need}")
    csv_dir = data_csv.parent

    rows: List[Tuple[str, str, float, int, str, str]] = []
    poppler = args.poppler_path or None
    soffice = args.soffice_path or None

    for _, r in tqdm(df.iterrows(), total=len(df), desc="infer"):
        page_id = str(r["ID"])
        raw_path = str(r["path"])
        file_path = (csv_dir / raw_path).resolve()
        target_wh = (int(r["width"]), int(r["height"]))

        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        try:
            images = convert_to_images(file_path, dpi=args.dpi, poppler_path=poppler, soffice_path=soffice)
        except Exception as e:
            print(f"[WARN] Convert failed: {file_path} -> {e}")
            continue

        for img in images:
            dets = detect_on_pil(model, img, args.imgsz, args.conf, args.iou, args.max_det, args.device, target_wh, ocr)
            dets.sort(key=lambda d: d.order if d.order >= 0 else 10**9)
            for d in dets:
                rows.append(d.as_row(page_id))

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "category_type", "confidence_score", "order", "text", "bbox"])
        w.writerows(rows)
    print(f"[OK] wrote {len(rows)} rows -> {out_csv}")

    # 옵션: 자동 시각화
    if args.auto_viz:
        viz_py = BASE_DIR / "viz_boxes_poppler.py"
        if viz_py.exists():
            out_dir = BASE_DIR / "viz"; out_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    ["python", str(viz_py),
                     "--data_csv", str(data_csv),
                     "--pred_csv", str(out_csv),
                     "--out_dir", str(out_dir)],
                    check=True
                )
                print(f"[OK] viz saved -> {out_dir}")
            except Exception as e:
                print(f"[WARN] viz failed: {e}")


if __name__ == "__main__":
    main()

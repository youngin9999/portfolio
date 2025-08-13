#!/usr/bin/env python3
r"""
YOLO + EasyOCR + PPTX(poppler) 지원
- PPTX: LibreOffice(headless)로 PDF 변환 → pdf2image(poppler)로 이미지 렌더
- PDF: pdf2image(poppler)로 렌더
- JPG/PNG: 그대로 로드
python yoloeasyocr.py ^
  --weights .\model\yolov11n-doclaynet.pt ^
  --data_csv .\data\test.csv ^
  --out_csv .\output\submission.csv ^
  --device 0 ^
  --poppler_path "C:\Contest\Release-24.08.0-0\poppler-24.08.0\Library\bin" ^
  --soffice_path "C:\Program Files\LibreOffice\program\soffice.exe"
 
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO

# PyMuPDF (일부 상황에서 보조로 쓸 수도 있음)
try:
    import pymupdf as fitz  # noqa: F401
except Exception:
    try:
        import fitz  # type: ignore # noqa: F401
    except Exception:
        fitz = None  # 선택적

# EasyOCR
import easyocr
# pdf2image(poppler)
from pdf2image import convert_from_path

# =====================
# Config / Constants
# =====================
CATEGORY_NAMES = ["title", "subtitle", "text", "image", "table", "equation"]
TEXT_CATS = {"title", "subtitle", "text"}

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
    return YOLO(str(weights))

def init_easyocr(langs: str = "ko,en", model_dir: str = "./easyocr_models",
                 download: bool = False, gpu: bool = False) -> easyocr.Reader:
    lang_list = [x.strip() for x in langs.split(",") if x.strip()]
    return easyocr.Reader(
        lang_list,
        download_enabled=download,
        model_storage_directory=model_dir,
        gpu=gpu
    )

def _find_soffice(soffice_path: str | None) -> str | None:
    if soffice_path:
        p = Path(soffice_path)
        if p.exists():
            return str(p)
    # 리눅스: soffice / libreoffice
    for name in ("soffice", "libreoffice"):
        exe = shutil.which(name)
        if exe:
            return exe
    # 윈도우 기본 경로 추정
    for p in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        if Path(p).exists():
            return p
    return None

def pdf_to_images_pdf2image(pdf_path: Path, dpi: int, poppler_path: str | None):
    kw = {"dpi": dpi}
    if poppler_path:
        kw["poppler_path"] = poppler_path
    # output_folder/ fmt 생략: 메모리 내 PIL 리스트 반환
    return convert_from_path(str(pdf_path), **kw)

def convert_to_images(input_path: Path, dpi: int = 200,
                      poppler_path: str | None = None,
                      soffice_path: str | None = None) -> List[Image.Image]:
    """
    - PDF : pdf2image(poppler)
    - PPTX: LibreOffice(headless)로 PDF 변환 후 pdf2image(poppler)
    - JPG/PNG: 바로 로드
    """
    ext = input_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return [Image.open(str(input_path)).convert("RGB")]
    if ext == ".pdf":
        return pdf_to_images_pdf2image(input_path, dpi=dpi, poppler_path=poppler_path)
    if ext == ".pptx":
        exe = _find_soffice(soffice_path)
        if not exe:
            raise RuntimeError(
                "LibreOffice(soffice) 실행파일을 찾지 못했습니다. "
                "리눅스는 'libreoffice' 패키지 또는 AppImage, 윈도는 soffice.exe 경로를 --soffice_path로 넘기세요."
            )
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
                raise RuntimeError(f"LibreOffice PDF 변환 실패: {input_path.name}")
            return pdf_to_images_pdf2image(pdf_path, dpi=dpi, poppler_path=poppler_path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

def run_ocr_text(reader: easyocr.Reader, image_pil: Image.Image,
                 bbox_xyxy: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    crop = np.array(image_pil.crop((x1, y1, x2, y2)))  # HWC RGB
    try:
        texts = reader.readtext(crop, detail=0)
        return " ".join([t for t in texts if isinstance(t, str)]).strip()
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
# Inference core
# =====================

def detect_on_pil(model: YOLO, image_pil: Image.Image, imgsz: int, conf: float, iou: float,
                  max_det: int, device: str | int, target_wh: Tuple[int, int],
                  ocr_engine: Any) -> List[Det]:
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
            det.text = run_ocr_text(ocr_engine, image_pil, (x1, y1, x2, y2))
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
    p.add_argument("--device", default="cpu")  # 'cpu' or 0 / 'cuda:0'
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=3000)

    # 렌더 옵션
    p.add_argument("--dpi", type=int, default=200, help="PDF/PPTX 렌더 DPI")
    p.add_argument("--poppler_path", default="", help="Windows에서 poppler bin 경로 (리눅스는 보통 필요 없음)")
    p.add_argument("--soffice_path", default="", help="LibreOffice soffice 실행파일 경로(없으면 PATH에서 탐색)")

    # EasyOCR 옵션
    p.add_argument("--easyocr_langs", default="ko,en")
    p.add_argument("--easyocr_model_dir", default="./easyocr_models")
    p.add_argument("--easyocr_download", action="store_true", help="첫 실행시 가중치 다운로드 허용(온라인)")
    p.add_argument("--easyocr_gpu", action="store_true", help="EasyOCR에서 GPU 사용")

    args = p.parse_args()

    model = load_yolo(Path(args.weights))
    ocr = init_easyocr(
        langs=args.easyocr_langs,
        model_dir=args.easyocr_model_dir,
        download=args.easyocr_download,
        gpu=bool(args.easyocr_gpu),
    )

    data_csv = Path(args.data_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    need_cols = {"ID", "path", "width", "height"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"{data_csv} must contain columns: {need_cols}")

    rows: List[Tuple[str, str, float, int, str, str]] = []
    csv_dir = data_csv.parent
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
            images = convert_to_images(file_path, dpi=args.dpi,
                                       poppler_path=poppler, soffice_path=soffice)
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

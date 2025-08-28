#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolopad.py — YOLO(레이아웃) + PaddleOCR(전체 페이지) → 병합 → submission.csv

사용법(예):
  python3 yolopad.py \
    --weights model/yolo/doclayout_yolo_docstructbench_imgsz1024.pt \
    --data_csv data/test.csv \
    --out_csv output/submission.csv \
    --imgsz 896 --device cpu --dpi 400

입력 CSV 형식:
  ID,path,width,height
출력 CSV 형식(대회 제출 포맷):
  ID,category_type,confidence_score,order,text,bbox

특징:
- PaddleOCR을 전체 이미지에 한 번만 실행(텍스트라인 폴리곤+문자열 획득)
- YOLO로 레이아웃(Text/Title/...) 박스 감지
- 중심점 포함 → IoU → 거리 기준으로 라인을 영역에 매칭해 텍스트 병합
- 한국어+영어 혼합 문장을 정확하게 처리(사용자 선호 모델 고정)
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
from typing import Any, Dict, List, Tuple

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

# OCR: PaddleOCR
from paddleocr import PaddleOCR

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
NON_TEXT_CATS = {"image", "table", "equation"}
BASE_DIR = Path(__file__).resolve().parent

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


# ===================== OCR (PaddleOCR Full-page) =====================
@dataclass
class PaddleOCREngine:
    ocr: PaddleOCR

def init_ocr_paddle(
    # 사용자가 고정한 한국어 v5 모바일 인식 모델(한/영/숫자 지원)
    text_recognition_model_name: str = "korean_PP-OCRv5_mobile_rec",
    text_recognition_model_dir: str = str(BASE_DIR / "model/ocr/korean_PP-OCRv5_mobile_rec_infer"),
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = True,
) -> PaddleOCREngine:
    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        text_recognition_model_name=text_recognition_model_name,
        text_recognition_model_dir=text_recognition_model_dir,
        # 보정 모듈 설정
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )
    return PaddleOCREngine(ocr=ocr)

def paddle_ocr_full(engine: PaddleOCREngine, image_pil: Image.Image) -> List[Dict[str, Any]]:
    """
    전체 이미지에서 텍스트 라인 폴리곤/텍스트/점수 추출
    반환: [{"text":str,"score":float,"poly":np.ndarray(4,2),"bbox":(x1,y1,x2,y2),"cx":..,"cy":..}, ...]
    """
    arr = np.array(image_pil.convert("RGB"))
    result = engine.ocr.predict(input=arr)
    # 디버그용 출력은 필요 시 유지
    # for res1 in result:
    #     res1.print(); res1.save_to_img("output"); res1.save_to_json("output")

    if not result:
        return []
    res = result[0]

    lines: List[Dict[str, Any]] = []

    # ── (A) PaddleOCR 3.x 표준 포맷: res.json['res'] 에서 파싱 ──
    data = getattr(res, "json", None)
    if isinstance(data, dict):
        data = data.get("res", data)

        rec_texts  = data.get("rec_texts")   # list[str]
        rec_scores = data.get("rec_scores")  # list/np.array[float]
        rec_polys  = data.get("rec_polys")   # np.array [N,4,2] (있으면 이걸 우선)
        rec_boxes  = data.get("rec_boxes")   # np.array [N,4]   (fallback)

        if rec_texts is not None:
            rec_scores = np.asarray(rec_scores).tolist() if rec_scores is not None else [0.0]*len(rec_texts)

            if rec_polys is not None:
                # 다각형 기반 (권장)
                polys = np.asarray(rec_polys, dtype=float)  # (N,4,2)
                for txt, score, poly in zip(rec_texts, rec_scores, polys):
                    txt = (txt or "").strip()
                    if not txt: continue
                    x1, y1 = float(poly[:,0].min()), float(poly[:,1].min())
                    x2, y2 = float(poly[:,0].max()), float(poly[:,1].max())
                    lines.append({
                        "text": txt, "score": float(score), "poly": poly.astype(float),
                        "bbox": (x1, y1, x2, y2), "cx": float(poly[:,0].mean()), "cy": float(poly[:,1].mean())
                    })
                return lines

            if rec_boxes is not None:
                # 사각 박스 기반 (폴리곤 없을 때 대체)
                boxes = np.asarray(rec_boxes, dtype=float)  # (N,4) 가정: [x1,y1,x2,y2]
                for txt, score, box in zip(rec_texts, rec_scores, boxes):
                    txt = (txt or "").strip()
                    if not txt: continue
                    x1, y1, x2, y2 = map(float, box)
                    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                    poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=float)
                    lines.append({
                        "text": txt, "score": float(score), "poly": poly,
                        "bbox": (x1, y1, x2, y2), "cx": cx, "cy": cy
                    })
                return lines
        # 여기까지 왔는데도 못 꺼냈다면 (B)로 폴백

    # ── (B) 다른 포맷(예: res.ocr_info / 구식 list 포맷) 폴백 ──
    if hasattr(res, "ocr_info") and res.ocr_info:
        for it in res.ocr_info:
            txt = (it.get("text") or it.get("transcription") or "").strip()
            if not txt: continue
            score = float(it.get("score", 0.0))
            poly  = it.get("box") or it.get("points")
            if not poly: continue
            poly = np.array(poly, dtype=float)
            x1, y1 = float(poly[:,0].min()), float(poly[:,1].min())
            x2, y2 = float(poly[:,0].max()), float(poly[:,1].max())
            lines.append({
                "text": txt, "score": score, "poly": poly,
                "bbox": (x1, y1, x2, y2), "cx": float(poly[:,0].mean()), "cy": float(poly[:,1].mean())
            })
        return lines

    # ── (C) 구버전 legacy 리스트 포맷 폴백 ──
    if isinstance(res, list):  # e.g., [[points, (text, conf)], ...]
        for line in res:
            if not (isinstance(line, list) and len(line) > 1):
                continue
            txt, conf = str(line[1][0]).strip(), float(line[1][1])
            pts = np.array(line[0], dtype=float)
            if txt and pts.size:
                x1, y1 = float(pts[:,0].min()), float(pts[:,1].min())
                x2, y2 = float(pts[:,0].max()), float(pts[:,1].max())
                lines.append({
                    "text": txt, "score": conf, "poly": pts,
                    "bbox": (x1, y1, x2, y2), "cx": float(pts[:,0].mean()), "cy": float(pts[:,1].mean())
                })
        return lines

    return lines


# ===================== Helpers: IoU / Center / Distance =====================


# ===================== Detection (YOLO) =====================

def load_model(weights: Path):
    model = _YOLO(str(weights))
    print(f"[INFO] backend={_BACKEND}, weights={weights}")
    return model

def detect_regions(
    model: Any,
    image_pil: Image.Image,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> Tuple[List[Det], Tuple[int,int]]:
    """YOLO로 레이아웃 영역 감지 → Det 리스트와 (W,H) 반환 (박스는 현재 이미지 좌표계)"""
    img_np = np.array(image_pil)  # RGB
    res = model.predict(img_np, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]

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
        x1, y1, x2, y2 = map(int, box.tolist())
        out.append(Det(name, float(cf), x1, y1, x2, y2, text=""))
    return out, (int(W), int(H))


# ===================== Merge OCR Lines into YOLO Regions =====================
def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    ba = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    return inter / max(1e-6, aa + ba - inter)

def _coverage(line_bbox, reg_bbox):
    # 라인 박스가 영역(reg)에 '얼마나' 들어가 있는지 (line 기준)
    ax1, ay1, ax2, ay2 = line_bbox; bx1, by1, bx2, by2 = reg_bbox
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    la = max(1e-6, (ax2-ax1) * (ay2-ay1))
    return inter / la

def _center_in(reg_bbox, cx, cy, margin=0):
    x1, y1, x2, y2 = reg_bbox
    return (cx >= x1 - margin) and (cx <= x2 + margin) and (cy >= y1 - margin) and (cy <= y2 + margin)

def merge_ocr_with_layout(
    lines: List[Dict[str, Any]],
    dets: List["Det"],
    *,
    center_margin: int = 6,
    coverage_thr: float = 0.50,
    iou_thr: float = 0.20,
    y_bucket: int = 8
) -> None:
    print(f"[MERGE] dets={len(dets)}, lines={len(lines)}", flush=True)
    if not dets or not lines:
        return

    # 비텍스트 박스는 확실히 비우기
    for d in dets:
        if d.cls_name in NON_TEXT_CATS:
            d.text = ""

    # 텍스트 박스만 순회
    for d in dets:
        if d.cls_name not in TEXT_CATS:
            continue

        reg = (d.x1, d.y1, d.x2, d.y2)
        assigned = []

        for ln in lines:
            lb = ln.get("bbox")
            if not lb or "cx" not in ln or "cy" not in ln or "text" not in ln:
                continue
            lcx, lcy = float(ln["cx"]), float(ln["cy"])

            # (a) 중심 포함(+동적 마진: 라인 높이 0.5×)
            lh = max(1.0, lb[3] - lb[1])
            dyn_margin = max(center_margin, int(lh * 0.5))
            ok = _center_in(reg, lcx, lcy, margin=dyn_margin)

            # (b) 라인 포함율
            if not ok and _coverage(lb, reg) >= coverage_thr:
                ok = True

            # (c) IoU
            if not ok and _iou(lb, reg) >= iou_thr:
                ok = True

            if ok:
                assigned.append(ln)

        print(f"[MERGE] det({d.cls_name}) {reg} -> assign {len(assigned)} lines", flush=True)

        if not assigned:
            d.text = ""
            continue

        assigned.sort(key=lambda t: (round(float(t["cy"]) / y_bucket) * y_bucket, float(t["cx"])))
        merged = " ".join([t["text"].strip() for t in assigned if t.get("text") and t["text"].strip()])
        d.text = merged.strip()
        print(f"[MERGE] det({d.cls_name}) text_len={len(d.text)}", flush=True)

    
# ===================== Misc Utils =====================
def scale_bbox_to_target(bbox_xyxy: Tuple[int,int,int,int], curr_wh: Tuple[int, int], target_wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
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
    """두 단 판별(대략) 후 y→x로 전체 순서 부여"""
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


# ===================== Main =====================
def main():
    default_weights = BASE_DIR / "model" / "yolo" / "doclayout_yolo_docstructbench_imgsz1024.pt"
    default_data = BASE_DIR / "data" / "test.csv"
    default_out  = BASE_DIR / "output" / "submission.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(default_weights))
    ap.add_argument("--data_csv", default=str(default_data))
    ap.add_argument("--out_csv",  default=str(default_out))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.6)
    ap.add_argument("--max_det", type=int, default=3000)

    # 렌더 옵션
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--poppler_path", default="")
    ap.add_argument("--soffice_path", default="")

    args = ap.parse_args()

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True) # csv 경로설정

    # 모델 / OCR
    model = load_model(Path(args.weights))
    ocr_engine = init_ocr_paddle(
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        text_recognition_model_dir= str(BASE_DIR / "model/ocr/korean_PP-OCRv5_mobile_rec_infer"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

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
            # 1) 전체 페이지 OCR (한 번만)
            lines = paddle_ocr_full(ocr_engine, img)
            print(f"lines = {lines}")                            ### 나중에 주석처리 << 여기까지 정상작동 확인 
            # 2) YOLO 레이아웃 감지
            dets, curr_wh = detect_regions(model, img, args.imgsz, args.conf, args.iou, args.max_det)
            # print(dets)                                          ###### 이것도 확인 후 주석처리
            # 3) 라인 ↔ 영역 병합


            merge_ocr_with_layout( lines, dets)
            print(f"=====final dets1 = {dets}")
            # 4) 순서 부여(두 단 고려)
            assign_reading_order(dets, page_w=curr_wh[0])
            print(f"=====final dets2 = {dets}")
            # 5) 출력(제출 포맷) — bbox를 타깃 해상도로 스케일
            for d in dets:
                X1, Y1, X2, Y2 = scale_bbox_to_target((d.x1, d.y1, d.x2, d.y2), curr_wh, target_wh)
                X1, Y1, X2, Y2 = clamp_bbox(X1, Y1, X2, Y2, *target_wh)
                bbox_str = f"{X1}, {Y1}, {X2}, {Y2}"
                rows.append((page_id, d.cls_name, round(float(d.conf), 6), int(d.order if d.order >= 0 else 0), d.text, bbox_str))
            print(f"rows = {rows}")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "category_type", "confidence_score", "order", "text", "bbox"])
        w.writerows(rows)
    print(f"[OK] wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()

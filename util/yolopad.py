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
_LABEL_PRI = {"title": 3, "subtitle": 2, "text" : 1}
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

# ================중복제거 , yolo 박스병합 =====================
def dedup_text_overlaps(
    dets: list,
    iou_thr: float = 0.90,      # 거의 같은 위치일 때 IoU 기준
    cover_thr: float = 0.95,    # 상호 포괄 비율(라인 기준 coverage) 기준
    center_margin: int = 6      # 중심 상호 포함 여유(px)
) -> list:
    """
    같은 위치로 중복된 텍스트계 박스(title/subtitle/text)는 1개만 남긴다.
    - 기준: IoU≥iou_thr OR (A in B ≥cover_thr AND B in A ≥cover_thr) OR 중심 상호 포함
    - 선택: 라벨 우선순위(title>subtitle>text) → conf 큰 것
    - 이미지/표/수식은 건드리지 않음
    - NEW: 부분 겹침(둘 다 텍스트 일부만 감싸는 상황)은 '합집합 박스'로 병합
    """
    if not dets:
        return dets
    
    # 텍스트/비텍스트 분리
    idx_text = [i for i,d in enumerate(dets) if d.cls_name in TEXT_CATS]
    idx_other = [i for i,d in enumerate(dets) if d.cls_name not in TEXT_CATS]

    # 우선순위→conf 내림차순으로 정렬(강한 것부터 살리고 뒤를 지움)
    order = sorted(
        idx_text,
        key=lambda i: (_LABEL_PRI.get(dets[i].cls_name, 0), float(dets[i].conf)),
        reverse=True
    )

    keep = [True]*len(dets)

    def _xyxy(d):
        return (float(d.x1), float(d.y1), float(d.x2), float(d.y2))

    for i, a_idx in enumerate(order):
        if not keep[a_idx]:
            continue

        # A는 루프 내에서 'Union으로 확장'될 수 있으므로,
        # 매번 최신 dets[a_idx]에서 좌표/중심을 다시 읽는다.  # NEW
        A = _xyxy(dets[a_idx])

        for b_idx in order[i+1:]:
            if not keep[b_idx]:
                continue

            B = _xyxy(dets[b_idx])

            # 같은 위치 판정: IoU or 상호 coverage or 중심 상호 포함
            same_place = False
            if _iou(A, B) >= iou_thr:
                same_place = True
            else:
                cab = _coverage(A, B)  # A가 B에 얼마나 들어가나
                cba = _coverage(B, A)  # B가 A에 얼마나 들어가나
                if cab >= cover_thr and cba >= cover_thr:
                    same_place = True
                else:
                    # 중심 상호 포함
                    acx = (A[0]+A[2])/2.0; acy = (A[1]+A[3])/2.0
                    bcx = (B[0]+B[2])/2.0; bcy = (B[1]+B[3])/2.0
                    if _center_in(A, bcx, bcy, margin=center_margin) and _center_in(B, acx, acy, margin=center_margin):
                        same_place = True

            if same_place:
                
                dets[a_idx].x1 = min(A[0], B[0])
                dets[a_idx].y1 = min(A[1], B[1])
                dets[a_idx].x2 = max(A[2], B[2])
                dets[a_idx].y2 = max(A[3], B[3])

                # 더 높은 우선순위 라벨(title>subtitle>text) 및 더 큰 conf 채택
                if _LABEL_PRI.get(dets[b_idx].cls_name, 0) > _LABEL_PRI.get(dets[a_idx].cls_name, 0):
                    dets[a_idx].cls_name = dets[b_idx].cls_name
                dets[a_idx].conf = max(float(dets[a_idx].conf), float(dets[b_idx].conf))

                keep[b_idx] = False

                # 이후 비교를 위해 A를 최신 Union 박스로 갱신
                A = _xyxy(dets[a_idx])
                continue


    # 결과 구성: 텍스트는 keep만, 비텍스트는 전부 유지
    out = []
    for i, d in enumerate(dets):
        if d.cls_name in TEXT_CATS:
            if keep[i]:
                out.append(d)
        else:
            out.append(d)
    title_idx = [i for i, d in enumerate(out) if getattr(d, "cls_name", "") == "title"]
    if len(title_idx) > 1:
        top_i = min(title_idx, key=lambda i: float(out[i].y1))
        for i in title_idx:
            if i != top_i:
                out[i].cls_name = "subtitle"
    
    return out

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
    center_margin: int = 6,    # 중심 포함 판정 기본 마진(px)
    coverage_thr: float = 0.50,# 라인 포함율 임계
    iou_thr: float = 0.20,     # IoU 임계
    y_bucket: int = 8          # 같은 줄 그룹핑 및 정렬 버킷(px)
) -> None:
    """
    1) dets 하나 고름 → 여기에 속한 lines를 찾아 병합.
       - 텍스트 박스(TEXT_CATS): d.text에 병합 결과 할당
       - 비텍스트: 배정만 체크(텍스트는 채우지 않음)
    2) 모든 dets에 대해 반복
    3) 어떤 det에도 속하지 않은 OCR 라인은 줄 단위로 묶어 새 Det(text)로 승격 후 병합
    """
    if not lines:
        return
    if dets is None:
        dets = []

    dets[:] = [
        d for d in dets
        if not (str(d.cls_name).lower() == "table" and float(getattr(d, "conf", 0.0)) <= 0.5) # 0.5 table conf thr
    ]

    # 준비
    assigned_flags = [False] * len(lines)   # OCR 라인이 어느 det에든 배정됐는지
    # 비텍스트는 텍스트 비워두기(명시적으로)
    for d in dets:
        if d.cls_name in NON_TEXT_CATS:
            d.text = ""

    # ── (1)(2) dets 순회: 배정 + (텍스트면) 병합 ──────────────────────────────
    for d in dets:
        reg = (float(d.x1), float(d.y1), float(d.x2), float(d.y2))
        assigned_idx: List[int] = []

        # 어떤 라인이 이 det에 속하는가?
        for j, ln in enumerate(lines):
            lb = ln["bbox"]                    # (x1,y1,x2,y2)
            lcx, lcy = float(ln["cx"]), float(ln["cy"])
            # 동적 마진(라인 높이 기반)으로 중심 포함 판정 완화
            lh = max(1.0, float(lb[3]) - float(lb[1]))
            dyn = max(center_margin, int(lh * 0.5))

            ok = _center_in(reg, lcx, lcy, margin=dyn)
            # if not ok and _coverage(lb, reg) >= coverage_thr: ok = True
            # if not ok and _iou(lb, reg)       >= iou_thr:     ok = True

            if ok:
                assigned_idx.append(j)
                assigned_flags[j] = True

        # 텍스트 박스만 병합해서 d.text 채움, 비텍스트는 배정 체크만
        if d.cls_name in TEXT_CATS:
            if not assigned_idx:
                d.text = ""
                continue
            # (y→x) 정렬 후 병합
            seg = [lines[j] for j in assigned_idx]
            seg.sort(key=lambda t: (round(float(t["cy"]) / y_bucket) * y_bucket, float(t["cx"])))
            merged = " ".join(
                (t["text"] or "").strip()
                for t in seg
                if (t.get("text") or "").strip()
            ).strip()
            d.text = merged
        else:
            # image/table/equation 등은 텍스트를 채우지 않음(요청 사양)
            # d.text = ""  # 이미 위에서 비워둠
            pass

    # ── (4) 어떤 det에도 속하지 않은 OCR 라인 → 줄 단위로 승격/병합 ─────────────
    from collections import defaultdict
    import numpy as np

    orphan_idx = [i for i, f in enumerate(assigned_flags)
                  if not f and (lines[i].get("text") or "").strip()]

    if orphan_idx:
        rows = defaultdict(list)
        for i in orphan_idx:
            key = int(round(float(lines[i]["cy"]) / y_bucket))
            rows[key].append(i)

        for _, idxs in rows.items():
            # 같은 줄 안에서 x 오름차순
            idxs.sort(key=lambda i: float(lines[i]["cx"]))

            # Union bbox
            x1 = min(lines[i]["bbox"][0] for i in idxs)
            y1 = min(lines[i]["bbox"][1] for i in idxs)
            x2 = max(lines[i]["bbox"][2] for i in idxs)
            y2 = max(lines[i]["bbox"][3] for i in idxs)
            # 텍스트 병합
            merged_text = " ".join(
                (lines[i]["text"] or "").strip()
                for i in idxs
                if (lines[i].get("text") or "").strip()
            ).strip()
            # 신뢰도(평균)
            scores = [float(lines[i].get("score", 0.5)) for i in idxs]
            conf = float(np.mean(scores)) if scores else 0.5

            dets.append(Det(
                cls_name="text",
                conf=conf,
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                text=merged_text,
                order=-1
            ))


    
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

            
            print(dets)                                          ###### 이것도 확인 후 주석처리
            # 3) 라인 ↔ 영역 병합
            
            dets = dedup_text_overlaps(dets, iou_thr=0.80, cover_thr=0.95, center_margin=6)

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

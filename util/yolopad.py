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

from doclayout_yolo import YOLOv10 as _YOLO
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
    ocr_version="PP-OCRv5",
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_detection_model_name="PP-OCRv5_server_det",
    text_detection_model_dir=str(BASE_DIR / "model/ocr/PP-OCRv5_server_det"),
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    text_recognition_model_dir=str(BASE_DIR / "model/ocr/korean_PP-OCRv5_mobile_rec_infer"),
) -> PaddleOCREngine:
    ocr = PaddleOCR(
        ocr_version=ocr_version,
        use_textline_orientation=use_textline_orientation,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        text_detection_model_name=text_detection_model_name,
        text_detection_model_dir=text_detection_model_dir,
        text_recognition_model_name=text_recognition_model_name,
        text_recognition_model_dir=text_recognition_model_dir,
    )
    return PaddleOCREngine(ocr=ocr)

def paddle_ocr_full(engine: PaddleOCREngine, image_pil: Image.Image) -> List[Dict[str, Any]]:
    """
    전체 이미지에서 텍스트 라인 폴리곤/텍스트/점수 추출
    반환: [{"text":str,"score":float,"poly":np.ndarray(4,2),"bbox":(x1,y1,x2,y2),"cx":..,"cy":..}, ...]
    """
    arr = np.array(image_pil.convert("RGB"))
    result = engine.ocr.predict(input=arr)
    # # 디버그용 출력은 필요 시 유지
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
    res = model.predict(img_np, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False , device='cuda')[0]

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
    coverage_thr: float = 0.50,# 라인 포함율 임계(지금은 미사용)
    iou_thr: float = 0.20,     # IoU 임계(지금은 미사용)
    y_bucket: int = 8,         # 같은 줄 그룹핑 및 정렬 버킷(px)
    table_conf_min: float = 0.50,  # table 저신뢰 제거 임계
) -> None:
    """
    1) dets 하나 고름 → 여기에 속한 lines를 찾아 배정.
       - 텍스트 박스(TEXT_CATS): d.text에 (y→x 정렬) 병합 결과 할당
       - 비텍스트: 배정만 체크(텍스트는 채우지 않음)
    2) 모든 dets 반복
    3) 어떤 det에도 속하지 않은 OCR 라인(orphan)을 줄 단위로 proto det 생성
    4) 기존 텍스트 det + proto det 대상으로 세로 엣지 맞닿음이면 Union 반복 병합
    5) 마지막에 proto det만 한 번에 dets.extend(proto_kept)
    """
    if not lines:
        return
    if dets is None:
        dets = []

    # (A) table 저신뢰 제거
    dets[:] = [
        d for d in dets
        if not (str(d.cls_name).lower() == "table" and float(getattr(d, "conf", 0.0)) <= table_conf_min)
    ]

    # 준비
    assigned_flags = [False] * len(lines)   # OCR 라인이 어느 det에든 배정됐는지
    for d in dets:
        if d.cls_name in NON_TEXT_CATS:
            d.text = ""   # 비텍스트는 텍스트 비워두기(명시적으로)

    # (1)(2) dets 순회: 배정 + (텍스트면) 병합
    for d in dets:
        reg = (float(d.x1), float(d.y1), float(d.x2), float(d.y2))
        assigned_idx: List[int] = []

        for j, ln in enumerate(lines):
            lb = ln["bbox"]                    # (x1,y1,x2,y2)
            lcx, lcy = float(ln["cx"]), float(ln["cy"])
            # 동적 마진(라인 높이 기반)으로 중심 포함 판정 완화
            lh = max(1.0, float(lb[3]) - float(lb[1]))
            dyn = max(center_margin, int(lh * 0.5))

            ok = _center_in(reg, lcx, lcy, margin=dyn)
            # 필요시 아래 두 줄을 되살리면 배정 범위를 넓힐 수 있음
            # if not ok and _coverage(lb, reg) >= coverage_thr: ok = True
            # if not ok and _iou(lb, reg)       >= iou_thr:     ok = True

            if ok:
                assigned_idx.append(j)
                assigned_flags[j] = True

        if d.cls_name in TEXT_CATS:
            if not assigned_idx:
                d.text = ""
                continue
            seg = [lines[j] for j in assigned_idx]
            seg.sort(key=lambda t: (round(float(t["cy"]) / y_bucket) * y_bucket, float(t["cx"])))
            merged = " ".join(
                (t["text"] or "").strip()
                for t in seg
                if (t.get("text") or "").strip()
            ).strip()
            d.text = merged
        else:
            # 비텍스트는 배정만 체크(텍스트 미할당)
            pass

    # (3) orphan OCR → 줄 단위 proto det 생성 (아직 dets에 append 안 함!)
    from collections import defaultdict

    orphan_idx = [i for i, f in enumerate(assigned_flags)
                  if not f and (lines[i].get("text") or "").strip()]

    proto: List[Dict[str, Any]] = []
    if orphan_idx:
        rows = defaultdict(list)
        for i in orphan_idx:
            key = int(round(float(lines[i]["cy"]) / y_bucket))
            rows[key].append(i)

        for _, idxs in rows.items():
            idxs.sort(key=lambda i: float(lines[i]["cx"]))  # 같은 줄 내 x 정렬 << y좌표 무조건 정렬 로직 바꿔야함
            x1 = min(lines[i]["bbox"][0] for i in idxs)
            y1 = min(lines[i]["bbox"][1] for i in idxs)
            x2 = max(lines[i]["bbox"][2] for i in idxs)
            y2 = max(lines[i]["bbox"][3] for i in idxs)
            merged_text = " ".join(
                (lines[i]["text"] or "").strip()
                for i in idxs
                if (lines[i].get("text") or "").strip()
            ).strip()
            scores = [float(lines[i].get("score", 0.5)) for i in idxs]
            conf = (sum(scores) / len(scores)) if scores else 0.5
            proto.append({
                "cls_name": "text",
                "conf": float(conf),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "text": merged_text,
                "order": -1,
                "_keep": True
            })

    # (4) 세로 병합: 기존 텍스트 det + proto det 대상으로 “위/아래 엣지 맞닿음”이면 Union
    #    - 기존 텍스트 det끼리도 병합 가능(뒤쪽 것을 제거)
    #    - 기존 vs proto가 붙으면 기존을 살리고 proto를 흡수

    changed = True
    while changed:
        changed = False

        # 풀 구성: (kind, ref/index, keep-flag)
        pool = []
        # existing text dets
        for idx, d in enumerate(dets):
            if d.cls_name in TEXT_CATS:
                pool.append({"kind": "exist", "idx": idx, "keep": True})
        # proto dets
        for k, p in enumerate(proto):
            if p["_keep"]:
                pool.append({"kind": "proto", "idx": k, "keep": True})

        # y1 오름차순
        def _y1_of(item):
            if item["kind"] == "exist":
                dd = dets[item["idx"]]; return float(dd.y1)
            else:
                pp = proto[item["idx"]]; return float(pp["y1"])
        pool.sort(key=_y1_of)

        for a in range(len(pool)):
            if not pool[a]["keep"]:
                continue
            # A 가져오기
            if pool[a]["kind"] == "exist":
                Aref = dets[pool[a]["idx"]]
                A = (Aref.x1, Aref.y1, Aref.x2, Aref.y2)
            else:
                Aref = proto[pool[a]["idx"]]
                A = (Aref["x1"], Aref["y1"], Aref["x2"], Aref["y2"])

            for b in range(a+1, len(pool)):
                if not pool[b]["keep"]:
                    continue

                if pool[b]["kind"] == "exist":
                    Bref = dets[pool[b]["idx"]]
                    B = (Bref.x1, Bref.y1, Bref.x2, Bref.y2)
                else:
                    Bref = proto[pool[b]["idx"]]
                    B = (Bref["x1"], Bref["y1"], Bref["x2"], Bref["y2"])

                if A[1]+A[3] > B[1]+B[3]:
                    share= ocrbbox_merge_check(B, A, eps=1.0, min_overlap=2.0)
                else :
                    share= ocrbbox_merge_check(A, B, eps=1.0, min_overlap=2.0)
                
                if not share:
                    continue

                # 누구를 남길까? 우선순위: existing > proto, 그다음 conf 큰 것
                def _score(obj):
                    if isinstance(obj, dict):
                        return float(obj["conf"])
                    else:
                        return float(obj.conf)

                keepA = False
                if (pool[a]["kind"] == "exist" and pool[b]["kind"] == "proto"):
                    keepA = True
                elif (pool[a]["kind"] == "proto" and pool[b]["kind"] == "exist"):
                    keepA = False
                else:
                    # 같은 종류면 conf 큰 쪽
                    keepA = _score(Aref) >= _score(Bref)

                # Union + 병합
                if keepA:
                    # A := A ∪ B, text join, conf=max
                    if isinstance(Aref, dict):
                        Aref["x1"] = min(Aref["x1"], B[0]); Aref["y1"] = min(Aref["y1"], B[1])
                        Aref["x2"] = max(Aref["x2"], B[2]); Aref["y2"] = max(Aref["y2"], B[3])
                        Aref["text"] = " ".join(t for t in [Aref["text"].strip(), (Bref.text if hasattr(Bref, "text") else Bref["text"]).strip()] if t).strip()
                        Aref["conf"] = max(float(Aref["conf"]), _score(Bref))
                    else:
                        Aref.x1 = min(Aref.x1, B[0]); Aref.y1 = min(Aref.y1, B[1])
                        Aref.x2 = max(Aref.x2, B[2]); Aref.y2 = max(Aref.y2, B[3])
                        Aref.text = " ".join(t for t in [Aref.text.strip(), (Bref["text"] if isinstance(Bref, dict) else Bref.text).strip()] if t).strip()
                        Aref.conf = max(float(Aref.conf), _score(Bref))
                    pool[b]["keep"] = False
                else:
                    # B := A ∪ B
                    if isinstance(Bref, dict):
                        Bref["x1"] = min(Bref["x1"], A[0]); Bref["y1"] = min(Bref["y1"], A[1])
                        Bref["x2"] = max(Bref["x2"], A[2]); Bref["y2"] = max(Bref["y2"], A[3])
                        Bref["text"] = " ".join(t for t in [Bref["text"].strip(), (Aref.text if hasattr(Aref, "text") else Aref["text"]).strip()] if t).strip()
                        Bref["conf"] = max(float(Bref["conf"]), _score(Aref))
                    else:
                        Bref.x1 = min(Bref.x1, A[0]); Bref.y1 = min(Bref.y1, A[1])
                        Bref.x2 = max(Bref.x2, A[2]); Bref.y2 = max(Bref.y2, A[3])
                        Bref.text = " ".join(t for t in [Bref.text.strip(), (Aref["text"] if isinstance(Aref, dict) else Aref.text).strip()] if t).strip()
                        Bref.conf = max(float(Bref.conf), _score(Aref))
                    pool[a]["keep"] = False

                changed = True
                # 갱신된 A/B 좌표 반영 위해 다음 비교로
                break  # b 루프 탈출 → a 인덱스부터 다시

        # pool 기준으로 삭제 반영
        if changed:
            # existing 중 keep=False 제거
            to_remove = {p["idx"] for p in pool if p["kind"]=="exist" and not p["keep"]}
            if to_remove:
                dets[:] = [d for idx, d in enumerate(dets) if idx not in to_remove]
            # proto의 keep 플래그도 반영
            for p in pool:
                if p["kind"]=="proto" and not p["keep"]:
                    proto[p["idx"]]["_keep"] = False

    # (5) 마지막에 proto det만 한 번에 append
    if proto:
        dets.extend([
            Det(cls_name=p["cls_name"], conf=p["conf"],
                x1=p["x1"], y1=p["y1"], x2=p["x2"], y2=p["y2"],
                text=p["text"], order=p["order"])
            for p in proto if p["_keep"]
        ])

    
# ===================== Misc Utils =====================
from typing import Optional

def ocrbbox_merge_check(
    A: Tuple[float,float,float,float],
    B: Tuple[float,float,float,float],
    *,
    eps: float = 1.0,          # 위/아래 엣지 정렬 허용 오차(px)
    min_overlap: float = 2.0   # 가로로 겹쳐야 하는 최소 길이(px). 0이면 모서리 접촉도 허용 A가 윗박스 , B가 아랫박스
) -> Tuple[bool, float, Optional[str]]:
    """
    두 박스가 '위아래로 붙어있는지'만 판정.
    반환: (share, overlap_len, relation)
      - share: 위/아래 엣지 공유 여부
      - overlap_len: 가로 방향 실제 겹친 길이(px)
      - relation: "A-bottom~B-top" 또는 "A-top~B-bottom" 또는 None
    """
    ax1, ay1, ax2, ay2 = A
    bx1, by1, bx2, by2 = B

    # 가로 방향 겹침 길이
    ox = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    if ox < min_overlap:
        return False

    # A의 아래 엣지 ≒ B의 위 엣지
    if ay2 > by1:
        return True


    return False

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



if __name__ == "__main__":
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
    ocr_version="PP-OCRv5",
    use_textline_orientation=False,  # ← 끄면 해당 모델 다운로드 자체가 없음
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_detection_model_name="PP-OCRv5_server_det",
    text_detection_model_dir=str(BASE_DIR / "model/ocr/PP-OCRv5_server_det"),
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    text_recognition_model_dir=str(BASE_DIR / "model/ocr/korean_PP-OCRv5_mobile_rec_infer"),
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
            # print(f"lines = {lines}")                            ### 나중에 주석처리 << 여기까지 정상작동 확인 
            # 2) YOLO 레이아웃 감지
            dets, curr_wh = detect_regions(model, img, args.imgsz, args.conf, args.iou, args.max_det)

            
            # print(dets)                                          ###### 이것도 확인 후 주석처리
            # 3) 중복제거 + title -> subtitle 강등
            
            dets = dedup_text_overlaps(dets, iou_thr=0.80, cover_thr=0.95, center_margin=6)

            # 4 )  dets , liens merge 하기  , lines -> dets 승격
            merge_ocr_with_layout( lines, dets)
            
            # print(f"=====final dets1 = {dets}")
            # 4) 순서 부여(두 단 고려)
            assign_reading_order(dets, page_w=curr_wh[0])
            # print(f"=====final dets2 = {dets}")
            # 5) 출력(제출 포맷) — bbox를 타깃 해상도로 스케일
            for d in dets:
                X1, Y1, X2, Y2 = scale_bbox_to_target((d.x1, d.y1, d.x2, d.y2), curr_wh, target_wh)
                X1, Y1, X2, Y2 = clamp_bbox(X1, Y1, X2, Y2, *target_wh)
                bbox_str = f"{X1}, {Y1}, {X2}, {Y2}"
                rows.append((page_id, d.cls_name, round(float(d.conf), 6), int(d.order if d.order >= 0 else 0), d.text, bbox_str))
            # print(f"rows = {rows}")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "category_type", "confidence_score", "order", "text", "bbox"])
        w.writerows(rows)
    print(f"[OK] wrote {len(rows)} rows -> {out_csv}")

#!/usr/bin/env python3
"""
viz_boxes.py
- submission.csv의 bbox를 test.csv가 가리키는 원본 문서 위에 시각화해서 PNG로 저장
- 이미지(png/jpg) + PDF(PyMuPDF로 렌더) 지원, PPTX는 스킵(사전 변환 필요)
- ID가 'ID_p2'처럼 페이지 접미사가 있으면 해당 페이지에만 그림. 없으면 모든 페이지에 그림.

사용 예시(프로젝트 루트):
  python viz_boxes.py --data_csv ./data/test.csv --pred_csv ./output/submission.csv --out_dir ./viz
  # 특정 ID만
  python viz_boxes.py --data_csv ./data/test.csv --pred_csv ./output/submission.csv --out_dir ./viz --ids TEST_01

의존성:
  pip install pandas pillow pymupdf numpy
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# PyMuPDF import (package name is 'pymupdf', import name is 'fitz')
try:
    import pymupdf as fitz
except Exception:
    import fitz  # type: ignore

# ------------------------
# 설정: 카테고리별 색상
# ------------------------
CLASS_COLORS = {
    "title":    (255,   0,   0),   # 빨강
    "subtitle": (255, 165,   0),   # 주황
    "text":     ( 50, 205,  50),   # 초록
    "image":    ( 30, 144, 255),   # 파랑
    "table":    (148,   0, 211),   # 보라
    "equation": (220,  20,  60),   # 진분홍/크림슨
}

PAGE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_p(?P<pi>\d+)$")  # ID_p2 → base=ID, pi=2


# ------------------------
# Helper 함수들
# ------------------------
def parse_id_page(id_str: str) -> Tuple[str, int | None]:
    """ID가 'ID_pN'이면 (base, N), 아니면 (ID, None)."""
    m = PAGE_SUFFIX_RE.match(id_str)
    if m:
        return m.group("base"), int(m.group("pi"))
    return id_str, None


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
        print(f"[SKIP] PPTX not supported. Convert to PDF/PNG first: {input_path.name}")
        return []
    else:
        print(f"[SKIP] Unsupported file type: {input_path}")
        return []


def parse_bbox_str(b: str) -> Tuple[int, int, int, int]:
    # "x1, y1, x2, y2" 또는 "x1 y1 x2 y2" 모두 허용
    s = b.replace(",", " ").split()
    if len(s) != 4:
        raise ValueError(f"Bad bbox format: {b!r}")
    x1, y1, x2, y2 = map(int, s)
    return x1, y1, x2, y2


def scale_bbox_from_target_to_img(
    bxyxy: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
    img_wh: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """submission.csv의 bbox(=test.csv의 width/height 좌표계)를 실제 이미지 크기로 변환."""
    x1, y1, x2, y2 = bxyxy
    tw, th = target_wh
    iw, ih = img_wh
    sx = iw / float(tw) if tw else 1.0
    sy = ih / float(th) if th else 1.0
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
    # clamp
    X1 = max(0, min(X1, iw - 1)); X2 = max(0, min(X2, iw - 1))
    Y1 = max(0, min(Y1, ih - 1)); Y2 = max(0, min(Y2, ih - 1))
    if X2 < X1: X1, X2 = X2, X1
    if Y2 < Y1: Y1, Y2 = Y2, Y1
    return X1, Y1, X2, Y2


def draw_box(draw: ImageDraw.ImageDraw, xyxy: Tuple[int, int, int, int], color: Tuple[int, int, int], width: int = 3):
    x1, y1, x2, y2 = xyxy
    for w in range(width):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)


def draw_label(draw: ImageDraw.ImageDraw, xyxy: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]):
    x1, y1, _, _ = xyxy
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textsize(label, font=font)
    top = max(0, y1 - th - 4)  # 화면 밖으로 나가지 않게
    # 배경 박스
    draw.rectangle([x1, top, x1 + tw + 6, top + th + 2], fill=color)
    draw.text((x1 + 3, top + 1), label, fill=(255, 255, 255), font=font)


def draw_legend(img: Image.Image, class_colors: dict, margin: int = 8, box_h: int = 18):
    """좌상단에 범례(legend) 그리기."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    x, y = margin, margin
    for cls, color in class_colors.items():
        draw.rectangle([x, y, x + box_h, y + box_h], fill=color)
        draw.text((x + box_h + 6, y), cls, fill=(0,0,0), font=font )
        y += box_h + 6


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ids", default="", help="Comma-separated list of IDs to visualize; empty=all")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--thickness", type=int, default=3)
    args = ap.parse_args()

    data_csv = Path(args.data_csv).resolve()
    pred_csv = Path(args.pred_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(data_csv)
    pred_df = pd.read_csv(pred_csv)

    # 필수 컬럼 확인
    for col in ["ID", "path", "width", "height"]:
        if col not in test_df.columns:
            raise ValueError(f"{data_csv} missing column: {col}")
    for col in ["ID", "category_type", "confidence_score", "order", "text", "bbox"]:
        if col not in pred_df.columns:
            raise ValueError(f"{pred_csv} missing column: {col}")

    # test.csv 기준 메타(파일경로, 타깃WH) 구성
    csv_dir = data_csv.parent
    meta: Dict[str, Tuple[Path, Tuple[int, int]]] = {}
    for _, r in test_df.iterrows():
        _id = str(r["ID"])
        raw_path = str(r["path"]).strip().replace("\\", "/")
        # 허용: './data/...'
        for prefix in ("./data/", "data/"):
            if raw_path.startswith(prefix):
                raw_path = raw_path[len(prefix):]
        fpath = (csv_dir / raw_path).resolve()
        meta[_id] = (fpath, (int(r["width"]), int(r["height"])))

    # 선택 ID 필터
    wanted = set([i.strip() for i in args.ids.split(",") if i.strip()]) if args.ids else None

    # 예측을 base ID 기준으로 그룹 (ID_p2 → base ID, page_idx 추출)
    pred_df["_base_id"] = pred_df["ID"].astype(str).apply(lambda s: parse_id_page(s)[0])
    pred_df["_page_idx"] = pred_df["ID"].astype(str).apply(lambda s: parse_id_page(s)[1])
    grouped = pred_df.groupby("_base_id")

    total_saved = 0
    for base_id, g in grouped:
        if wanted and base_id not in wanted:
            continue
        if base_id not in meta:
            print(f"[WARN] ID {base_id} not found in {data_csv}, skipping")
            continue

        fpath, tgt_wh = meta[base_id]
        if not fpath.exists():
            print(f"[WARN] Missing file for ID={base_id}: {fpath}")
            continue

        images = convert_to_images(fpath, dpi=args.dpi)
        if not images:
            continue

        for pi, img in enumerate(images, start=1):
            # 페이지 접미사가 있는 row만 그 페이지에 표시. 접미사가 없으면 모든 페이지에 표시.
            g_this = pd.concat(
                [g[g["_page_idx"].isna()], g[g["_page_idx"] == pi]],
                ignore_index=True,
            )

            draw = ImageDraw.Draw(img)
            iw, ih = img.size
            for _, pr in g_this.iterrows():
                cls = str(pr["category_type"]).strip().lower()
                color = CLASS_COLORS.get(cls, (255, 255, 0))  # 미정의 클래스는 노랑
                try:
                    bx = parse_bbox_str(str(pr["bbox"]))
                    x1, y1, x2, y2 = scale_bbox_from_target_to_img(bx, tgt_wh, (iw, ih))
                except Exception:
                    continue
                # 박스 & 라벨
                draw_box(draw, (x1, y1, x2, y2), color, width=args.thickness)
                try:
                    lbl = f"{cls} {float(pr['confidence_score']):.2f}"
                except Exception:
                    lbl = cls
                try:
                    draw_label(draw, (x1, y1, x2, y2), lbl, color)
                except Exception:
                    pass

            # 범례 추가
            draw_legend(img, CLASS_COLORS)

            out_name = f"{base_id}_p{pi}.png" if len(images) > 1 else f"{base_id}.png"
            out_path = out_dir / out_name
            img.save(out_path)
            total_saved += 1
            print(f"[SAVE] {out_path}")

    print(f"[DONE] saved {total_saved} annotated image(s) to {out_dir}")


if __name__ == "__main__":
    main()

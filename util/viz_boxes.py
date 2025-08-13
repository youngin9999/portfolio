# -*- coding: utf-8 -*-
r"""
viz_boxes_poppler.py
- submission.csv의 bbox를 test.csv 원본 위에 그려 PNG로 저장
- PDF 렌더링: pdf2image + Poppler (pdfinfo/pdftoppm 필요)
- PPTX: LibreOffice(soffice)로 PPTX→PDF 변환 후, pdf2image로 렌더
- 이미지(png/jpg)는 그대로 사용
- ID가 'ID_p2'면 해당 페이지에만 그리기 지원(없으면 모든 페이지)

필요 패키지: pandas, pillow, pdf2image, numpy
윈도우 필수: Poppler(bin 폴더에 pdfinfo.exe, pdftoppm.exe), LibreOffice(soffice.exe)

예시(CMD):
  python viz_boxes.py ^
    --data_csv .\data\test.csv ^
    --pred_csv .\output\submission.csv ^
    --out_dir .\viz ^
    --poppler_path "C:\Contest\Release-24.08.0-0\poppler-24.08.0\Library\bin"^
    --soffice_path "C:\Program Files\LibreOffice\program\soffice.exe"
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

# 카테고리별 색상
CLASS_COLORS = {
    "title":    (255,   0,   0),   # 빨강
    "subtitle": (255, 165,   0),   # 주황
    "text":     ( 50, 205,  50),   # 초록
    "image":    ( 30, 144, 255),   # 파랑
    "table":    (148,   0, 211),   # 보라
    "equation": (220,  20,  60),   # 진분홍
}

PAGE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_p(?P<pi>\d+)$")  # ID_p2 → base=ID, pi=2

def parse_id_page(id_str: str) -> Tuple[str, int | None]:
    """ID가 'ID_pN'이면 (base, N), 아니면 (ID, None)."""
    m = PAGE_SUFFIX_RE.match(id_str)
    if m:
        return m.group("base"), int(m.group("pi"))
    return id_str, None

def _find_soffice(user_path: str | None) -> str | None:
    """soffice.exe 경로 탐색: 인자 → PATH → 흔한 설치 경로."""
    if user_path:
        p = Path(user_path)
        if p.is_file():
            return str(p)
    exe = shutil.which("soffice")
    if exe:
        return exe
    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/usr/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
        "/snap/bin/libreoffice",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def pdf_to_images_pdf2image(pdf_path: Path, dpi: int, poppler_path: str | None) -> List[Image.Image]:
    """Poppler(pdfinfo/pdftoppm)로 PDF → PIL 이미지 리스트"""
    kwargs = dict(dpi=dpi, fmt="png")
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    return convert_from_path(str(pdf_path), **kwargs)

def pptx_to_pdf_via_soffice(pptx_path: Path, out_dir: Path, soffice_path: str) -> Path:
    """LibreOffice headless 로 PPTX→PDF 변환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [soffice_path, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice convert failed ({proc.returncode}): {proc.stderr[:300]}")
    expected = out_dir / (pptx_path.with_suffix(".pdf").name)
    if expected.exists():
        return expected
    pdfs = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        raise FileNotFoundError(f"PDF not found after soffice convert: {pptx_path}")
    return pdfs[0]

def convert_to_images(input_path: Path, dpi: int, poppler_path: str | None, soffice_path: str | None) -> List[Image.Image]:
    """원본 파일을 페이지별 PIL 이미지로 변환."""
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        return pdf_to_images_pdf2image(input_path, dpi, poppler_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        return [Image.open(str(input_path)).convert("RGB")]
    elif ext == ".pptx":
        soffice = _find_soffice(soffice_path)
        if not soffice:
            print(f"[SKIP] LibreOffice(soffice) not found. --soffice_path 지정 필요: {input_path.name}")
            return []
        with tempfile.TemporaryDirectory(prefix="pptx2pdf_") as tmpd:
            pdf_path = pptx_to_pdf_via_soffice(input_path, Path(tmpd), soffice)
            return pdf_to_images_pdf2image(pdf_path, dpi, poppler_path)
    else:
        print(f"[SKIP] Unsupported file type: {input_path}")
        return []

def parse_bbox_str(b: str) -> Tuple[int, int, int, int]:
    """'x1, y1, x2, y2' 또는 공백 구분 허용."""
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
    """submission.csv bbox(=test.csv width/height 좌표계)를 실제 이미지 크기로 스케일링."""
    x1, y1, x2, y2 = bxyxy
    tw, th = target_wh
    iw, ih = img_wh
    sx = iw / float(tw) if tw else 1.0
    sy = ih / float(th) if th else 1.0
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
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
    top = max(0, y1 - th - 4)
    draw.rectangle([x1, top, x1 + tw + 6, top + th + 2], fill=color)
    draw.text((x1 + 3, top + 1), label, fill=(255, 255, 255), font=font)

def draw_legend(img: Image.Image, class_colors: dict, margin: int = 8, box_h: int = 18):
    """좌상단 범례."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    x, y = margin, margin
    for cls, color in class_colors.items():
        draw.rectangle([x, y, x + box_h, y + box_h], fill=color)
        draw.text((x + box_h + 6, y), cls, fill=(0, 0, 0), font=font)
        y += box_h + 6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ids", default="", help="Comma-separated list of base IDs to visualize; empty=all")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--poppler_path", default="", help="Poppler bin 폴더 경로(안에 pdfinfo.exe, pdftoppm.exe)")
    ap.add_argument("--soffice_path", default="", help="LibreOffice soffice 실행 파일 경로")
    args = ap.parse_args()

    data_csv = Path(args.data_csv).resolve()
    pred_csv = Path(args.pred_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(data_csv)
    pred_df = pd.read_csv(pred_csv)

    # 필수 컬럼 체크
    for col in ["ID", "path", "width", "height"]:
        if col not in test_df.columns:
            raise ValueError(f"{data_csv} missing column: {col}")
    for col in ["ID", "category_type", "confidence_score", "order", "text", "bbox"]:
        if col not in pred_df.columns:
            raise ValueError(f"{pred_csv} missing column: {col}")

    # test.csv 기준 메타 구성 (원본 경로 + target width/height)
    csv_dir = data_csv.parent
    meta: Dict[str, Tuple[Path, Tuple[int, int]]] = {}
    for _, r in test_df.iterrows():
        _id = str(r["ID"])
        raw_path = str(r["path"]).strip().replace("\\", "/")
        for prefix in ("./data/", "data/"):  # 상대경로 정리
            if raw_path.startswith(prefix):
                raw_path = raw_path[len(prefix):]
        fpath = (csv_dir / raw_path).resolve()
        meta[_id] = (fpath, (int(r["width"]), int(r["height"])))

    wanted = set([i.strip() for i in args.ids.split(",") if i.strip()]) if args.ids else None

    # 예측을 base ID 기준 그룹 (ID_p2 → base ID, page_idx 추출)
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

        poppler = args.poppler_path or None
        soffice  = args.soffice_path or None

        try:
            images = convert_to_images(fpath, dpi=args.dpi, poppler_path=poppler, soffice_path=soffice)
        except Exception as e:
            print(f"[WARN] convert failed: {fpath.name} -> {e}")
            continue
        if not images:
            continue

        for pi, img in enumerate(images, start=1):
            iw, ih = img.size
            draw = ImageDraw.Draw(img)

            # 해당 페이지 row들만 (ID 접미사 없는 row는 모든 페이지에 그림)
            g_this = pd.concat(
                [g[g["_page_idx"].isna()], g[g["_page_idx"] == pi]],
                ignore_index=True,
            )

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

    print(f"[DONE] saved {total_saved} image(s) to {out_dir}")

if __name__ == "__main__":
    main()

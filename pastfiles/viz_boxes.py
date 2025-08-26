# -*- coding: utf-8 -*-
r"""
viz_boxes.py  — 삼성 경진대회용 시각화(경로 하드코딩판)
- 실행만 하면(util 폴더에서) data/test.csv, output/submission.csv를 읽고
  원본 문서 위에 bbox를 그려 util/viz/ 에 PNG 저장
- PDF: pdf2image + Poppler 사용 (poppler bin 경로 하드코딩 + 자동탐색)
- PPTX: LibreOffice(soffice.exe)로 PPTX→PDF 변환 후 렌더
- 이미지(png/jpg)는 그대로 사용
- ID가 'ID_p2'면 2페이지만 그림, 접미사 없으면 모든 페이지에 그림

필요 pip: pandas pillow pdf2image numpy (옵션: pymupdf; Poppler 실패 시 폴백)
"""

from __future__ import annotations
import os, re, shutil, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

# ───────────────── 기본 경로(네 폴더 구조에 맞춰 고정) ─────────────────
BASE_DIR = Path(__file__).resolve().parent         # C:\Contest\sam\util
DATA_CSV = BASE_DIR / "data" / "test.csv"
PRED_CSV = BASE_DIR / "output" / "submission.csv"
OUT_DIR  = BASE_DIR / "viz"

# Poppler / LibreOffice 경로(윈도우 하드코딩 + ENV + PATH 자동탐색)
POPPLER_BIN_GUESS = r"C:\Users\young\miniconda3\envs\doc6\Library\bin"
SOFFICE_EXE_GUESS = r"C:\Program Files\LibreOffice\program\soffice.exe"

# ───────────── 색/유틸 ─────────────
CLASS_COLORS = {
    "title":    (255,   0,   0),
    "subtitle": (255, 165,   0),
    "text":     ( 50, 205,  50),
    "image":    ( 30, 144, 255),
    "table":    (148,   0, 211),
    "equation": (220,  20,  60),
}
FALLBACK_COLOR = (255, 255, 0)
PAGE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_p(?P<pi>\d+)$")

def parse_id_page(s: str) -> Tuple[str, int | None]:
    m = PAGE_SUFFIX_RE.match(str(s))
    return (m.group("base"), int(m.group("pi"))) if m else (str(s), None)

def _pick_font(size=16):
    cands = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in cands:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_box(draw, xyxy, color, width=3):
    x1, y1, x2, y2 = xyxy
    for w in range(width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=color)

def draw_label(draw, xyxy, text, color):
    x1, y1, _, _ = xyxy
    font = _pick_font(16)
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    top = max(0, y1 - th - 4)
    draw.rectangle([x1, top, x1+tw+6, top+th+2], fill=color)
    draw.text((x1+3, top+1), text, fill=(255,255,255), font=font)

def draw_legend(img):
    draw = ImageDraw.Draw(img)
    font = _pick_font(14)
    x, y = 8, 8
    for k, c in CLASS_COLORS.items():
        draw.rectangle([x, y, x+18, y+18], fill=c)
        draw.text((x+24, y-1), k, fill=(0,0,0), font=font)
        y += 22

def parse_bbox_str(b: str) -> Tuple[int,int,int,int]:
    s = str(b).strip().replace("[","").replace("]","").replace("(","").replace(")","")
    parts = s.replace(",", " ").split()
    if len(parts) != 4: raise ValueError(f"Bad bbox: {b!r}")
    x1, y1, x2, y2 = map(lambda v: int(float(v)), parts)
    return x1, y1, x2, y2

def scale_bbox(bxyxy, target_wh, img_wh):
    x1, y1, x2, y2 = bxyxy
    tw, th = target_wh
    iw, ih = img_wh
    sx = iw / float(tw) if tw else 1.0
    sy = ih / float(th) if th else 1.0
    X1, Y1 = int(round(x1*sx)), int(round(y1*sy))
    X2, Y2 = int(round(x2*sx)), int(round(y2*sy))
    X1 = max(0, min(X1, iw-1)); X2 = max(0, min(X2, iw-1))
    Y1 = max(0, min(Y1, ih-1)); Y2 = max(0, min(Y2, ih-1))
    if X2 < X1: X1, X2 = X2, X1
    if Y2 < Y1: Y1, Y2 = Y2, Y1
    return X1, Y1, X2, Y2

# ───────────── Poppler/soffice 경로 탐색 ─────────────
def _pick_poppler_bin() -> str | None:
    for p in [os.getenv("POPPLER_PATH"), POPPLER_BIN_GUESS]:
        if p and Path(p).exists():
            return p
    # PATH에 있으면 None 넘겨도 pdf2image가 알아서 찾음
    return None

def _pick_soffice() -> str | None:
    for p in [os.getenv("SOFFICE_PATH"), SOFFICE_EXE_GUESS, shutil.which("soffice")]:
        if p and Path(p).exists():
            return p
    return None

# ───────────── 변환기 ─────────────
def pdf_to_images_poppler(pdf_path: Path, dpi=200) -> List[Image.Image]:
    poppler_bin = _pick_poppler_bin()
    kw = dict(dpi=dpi, fmt="png")
    if poppler_bin:
        kw["poppler_path"] = poppler_bin
    return convert_from_path(str(pdf_path), **kw)

def pdf_to_images_fitz(pdf_path: Path, dpi=200) -> List[Image.Image]:
    # Poppler 실패 시 폴백 (옵션: pip install pymupdf)
    try:
        import fitz  # pymupdf
    except Exception as e:
        raise RuntimeError("Poppler 실패 & PyMuPDF 미설치") from e
    pages = []
    zoom = dpi/72.0
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
    return pages

def pptx_to_pdf_via_soffice(pptx_path: Path, out_dir: Path) -> Path:
    soffice = _pick_soffice()
    if not soffice:
        raise FileNotFoundError("LibreOffice(soffice) 경로를 찾지 못했습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"soffice 변환 실패({proc.returncode}): {proc.stderr[:200]}")
    expected = out_dir / (pptx_path.with_suffix(".pdf").name)
    if expected.exists(): return expected
    pdfs = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs: raise FileNotFoundError("soffice 변환 후 PDF를 찾지 못했습니다.")
    return pdfs[0]

def file_to_images(input_path: Path, dpi=200) -> List[Image.Image]:
    ext = input_path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        return [Image.open(str(input_path)).convert("RGB")]
    if ext == ".pdf":
        try:
            return pdf_to_images_poppler(input_path, dpi=dpi)
        except Exception:
            return pdf_to_images_fitz(input_path, dpi=dpi)
    if ext == ".pptx":
        with tempfile.TemporaryDirectory(prefix="pptx2pdf_") as td:
            pdf_path = pptx_to_pdf_via_soffice(input_path, Path(td))
            try:
                return pdf_to_images_poppler(pdf_path, dpi=dpi)
            except Exception:
                return pdf_to_images_fitz(pdf_path, dpi=dpi)
    raise ValueError(f"지원하지 않는 파일 형식: {input_path}")

# ───────────── 메인 로직(인자 없이 실행) ─────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 로드 (UTF-8 BOM 대응)
    test_df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    pred_df = pd.read_csv(PRED_CSV, encoding="utf-8-sig")

    # 메타: ID → (파일경로, (w,h))
    csv_dir = DATA_CSV.parent
    meta: Dict[str, Tuple[Path, Tuple[int,int]]] = {}
    for _, r in test_df.iterrows():
        _id = str(r["ID"])
        raw = str(r["path"]).strip().replace("\\", "/")
        # './data/...' 또는 'data/...'를 util 기준 절대경로로 정규화
        if raw.startswith("./"): raw = raw[2:]
        if raw.startswith("data/"): fpath = BASE_DIR / raw
        else:                       fpath = (csv_dir / raw)
        meta[_id] = (fpath.resolve(), (int(r["width"]), int(r["height"])))

    # 페이지 접미사 파싱
    pred_df["_base"] = pred_df["ID"].astype(str).apply(lambda s: parse_id_page(s)[0])
    pred_df["_pi"]   = pred_df["ID"].astype(str).apply(lambda s: parse_id_page(s)[1])
    groups = pred_df.groupby("_base", sort=False)

    saved = 0
    for base_id, g in groups:
        if base_id not in meta:
            print(f"[WARN] test.csv에 없는 ID → {base_id}")
            continue
        fpath, tgt_wh = meta[base_id]
        if not fpath.exists():
            print(f"[WARN] 파일 없음 → {fpath}")
            continue

        # 원본을 페이지별 이미지로
        try:
            images = file_to_images(fpath, dpi=200)
        except Exception as e:
            print(f"[WARN] 변환 실패: {fpath.name} → {e}")
            continue
        if not images:
            continue

        # 각 페이지별로 박스 그리기
        for idx, img in enumerate(images, start=1):
            # 접미사 없는 행 + 해당 페이지 행
            gg = pd.concat([g[g["_pi"].isna()], g[g["_pi"] == idx]], ignore_index=True)
            draw = ImageDraw.Draw(img)
            iw, ih = img.size

            for _, pr in gg.iterrows():
                cls = str(pr["category_type"]).strip().lower()
                color = CLASS_COLORS.get(cls, FALLBACK_COLOR)
                try:
                    bx = parse_bbox_str(str(pr["bbox"]))
                    x1, y1, x2, y2 = scale_bbox(bx, tgt_wh, (iw, ih))
                except Exception:
                    continue
                draw_box(draw, (x1, y1, x2, y2), color, width=3)
                try:
                    lbl = f"{cls} {float(pr['confidence_score']):.2f}"
                except Exception:
                    lbl = cls
                try:
                    draw_label(draw, (x1, y1, x2, y2), lbl, color)
                except Exception:
                    pass

            draw_legend(img)
            out_name = f"{base_id}_p{idx}.png" if len(images) > 1 else f"{base_id}.png"
            out_path = OUT_DIR / out_name
            img.save(out_path)
            saved += 1
            print(f"[SAVE] {out_path}")

    print(f"[DONE] saved {saved} image(s) → {OUT_DIR}")

if __name__ == "__main__":
    main()

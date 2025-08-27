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
from pytesseract import Output
import cv2
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
_HANGUL_RE = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]')
_LATIN_RE  = re.compile(r'[A-Za-z]')

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
# ===== OCR ONLY (Tesseract, KO+EN 혼합 개선) =====

def _contains_hangul(s: str) -> bool: return bool(_HANGUL_RE.search(s))
def _contains_latin(s: str) -> bool:  return bool(_LATIN_RE.search(s))

@dataclass
class TesseractCfg:
    lang: str = "kor+eng"  # 기본 혼합
    oem: int = 3           # LSTM
    psm: int = 6           # 균일 블록 가정
    extra: str = ""        # 추가 옵션
    tesseract_cmd: str | None = None

def _langs_to_tesseract_code(langs: str) -> str:
    m = {"ko":"kor","kor":"kor","korean":"kor","en":"eng","eng":"eng","english":"eng"}
    parts = [p.strip().lower() for p in (langs or "eng").split(",") if p.strip()]
    return "+".join(m.get(p, p) for p in parts) if parts else "eng"

def init_ocr(langs="ko,en", tesseract_cmd="", oem=3, psm=6, extra="") -> TesseractCfg:
    cfg = TesseractCfg(lang=_langs_to_tesseract_code(langs), oem=int(oem), psm=int(psm), extra=str(extra))
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        cfg.tesseract_cmd = tesseract_cmd
    return cfg

# ---- 전처리(업스케일 + 적응형 이진화 + 약한 샤프닝)
def _preproc_pil(pil: Image.Image) -> Image.Image:
    img = np.array(pil)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    scale = 2.0 if max(h, w) < 800 else 1.5 if max(h, w) < 1400 else 1.0
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 7, 10, 10)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    bw = cv2.filter2D(bw, -1, k)
    return Image.fromarray(bw)

def _strip_spaces(s: str) -> str:
    return " ".join(s.split())

def _ocr_pass_image_to_string(crop: Image.Image, lang: str, oem: int, psm: int,
                              extra_cfg: str = "", whitelist: str | None = None) -> str:
    cfgs = [f"--oem {int(oem)}", f"--psm {int(psm)}", "-c preserve_interword_spaces=1"]
    if whitelist:
        cfgs.append(f"-c tessedit_char_whitelist={whitelist}")
    if extra_cfg:
        cfgs.append(extra_cfg)
    config = " ".join(cfgs)
    out = pytesseract.image_to_string(crop, lang=lang, config=config)
    return _strip_spaces(out)

def _ocr_data_words(crop: Image.Image, lang: str, oem: int, psm: int,
                    extra_cfg: str = "", whitelist: str | None = None):
    cfgs = [f"--oem {int(oem)}", f"--psm {int(psm)}", "-c preserve_interword_spaces=1"]
    if whitelist:
        cfgs += [f"-c tessedit_char_whitelist={whitelist}",
                 "-c load_system_dawg=0", "-c load_freq_dawg=0"]
    if extra_cfg:
        cfgs.append(extra_cfg)
    config = " ".join(cfgs)
    data = pytesseract.image_to_data(crop, lang=lang, config=config, output_type=Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt: continue
        conf = int(data["conf"][i]) if str(data["conf"][i]).isdigit() else -1
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({"text": txt, "conf": conf, "bbox": (x, y, x+w, y+h)})
    return words

def _fix_ascii_token(tok: str) -> str:
    letters = [c for c in tok if c.isalpha()]
    if not letters: return tok
    upp = sum(1 for c in letters if c.isupper())
    if upp / max(1, len(letters)) >= 0.6:
        tok = tok.replace('!', 'I')
        if re.search(r'[A-Za-z]0[A-Za-z]', tok): tok = tok.replace('0', 'O')
        tok = re.sub(r'(^|[^a-z])l(?=[A-Z]|$)', r'\1I', tok)
    return tok

def _key(w): 
        (x1, y1, x2, y2) = w["bbox"]; return (y1, x1)


def _merge_tokens(eng_words, kor_words) -> str:
    merged = []
    i = j = 0
    while i < len(eng_words) or j < len(kor_words):
        ew = eng_words[i] if i < len(eng_words) else None
        kw = kor_words[j] if j < len(kor_words) else None
        pick_from = "eng" if ew and (not kw or _key(ew) <= _key(kw)) else "kor"
        pick = ew if pick_from == "eng" else kw
        if not pick: break
        t = pick["text"]
        if _contains_hangul(t) and kw:
            t = kw["text"] if kw["conf"] >= (ew["conf"] if ew else -1) else t
        elif _contains_latin(t) and ew:
            t = _fix_ascii_token(ew["text"] if ew["conf"] >= (kw["conf"] if kw else -1) else t)
        merged.append(t)
        if pick_from == "eng": i += 1
        else: j += 1
    s = " ".join(merged)
    s = re.sub(r'\s+([,.:;)\]])', r'\1', s)
    s = re.sub(r'([(\[])\s+', r'\1', s)
    return s.strip()

def run_ocr_text(reader: TesseractCfg, image_pil: Image.Image, bbox_xyxy) -> str:
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    crop = image_pil.crop((x1, y1, x2, y2))
    try:
        crop = _preproc_pil(crop)
    except Exception:
        pass

    # 1) 혼합 한 번 (참고값)
    base = _ocr_pass_image_to_string(
        crop, lang=getattr(reader, "lang", "kor+eng") or "kor+eng",
        oem=getattr(reader, "oem", 3), psm=getattr(reader, "psm", 6),
        extra_cfg=getattr(reader, "extra", "") or ""
    )
    # 2) 영어/한국어 별도 패스(단어/신뢰도)
    eng_words = _ocr_data_words(
        crop, lang="eng", oem=getattr(reader, "oem", 3), psm=6,
        whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+&/#():%.,"
    )
    kor_words = _ocr_data_words(
        crop, lang="kor", oem=getattr(reader, "oem", 3), psm=6
    )
    mixed = _merge_tokens(eng_words, kor_words)

    has_h = _contains_hangul(base) or any(_contains_hangul(w["text"]) for w in kor_words)
    has_l = _contains_latin(base)  or any(_contains_latin(w["text"]) for w in eng_words)
    if has_h and has_l: return mixed or base
    if has_h:           return " ".join(w["text"] for w in kor_words) or base
    if has_l:           return " ".join(_fix_ascii_token(w["text"]) for w in eng_words) or base
    return mixed or base

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

# CUDA 12.8 + cuDNN + Ubuntu 22.04 (필요시 12.x의 다른 근접 태그로 교체)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    poppler-utils poppler-data \
    libreoffice ghostscript \
    tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    fonts-nanum locales ca-certificates curl git build-essential \
 && locale-gen ko_KR.UTF-8 && update-locale LANG=ko_KR.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python3 -m pip install --upgrade pip

# PyTorch/cu128 (전용 인덱스)
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128

# 나머지 파이썬 패키지 (기본 PyPI)
RUN python3 -m pip install \
    tensorflow==2.19.0 \
    pillow==11.0.0 \
    PyMuPDF==1.24.11 \
    pandas==2.0.3 \
    numpy==1.26.4 \
    faiss-gpu-cu12==1.11.0 \
    langchain==0.1.6 \
    pytesseract==0.3.13 \
    ultralytics==8.3.170 \
    scikit-learn==1.7.1 \
    bitsandbytes==0.46.1 \
    accelerate==1.9.0 \
    transformers==4.46.3 \
    python-pptx==0.6.23 \
    pdfminer.six==20221105 \
    opencv-python==4.9.0.80 \
    pdf2image==1.17.0 \
    sentencepiece==0.1.99 \
    regex==2023.12.25 \
    easyocr==1.7.1 \
    layoutparser==0.3.4 \
    tqdm==4.66.4 \
    loguru==0.7.2 \
    pyyaml==6.0.1 \
    rich==13.7.1 \
    albumentations==1.4.3

# 작업 디렉토리
WORKDIR /workspace

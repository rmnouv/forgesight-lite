# ForgeSight-Lite

> A two-stream (RGB + DCT) deep learning detector for image forgeries in financial
> and administrative documents, packaged as a Docker-ready REST API.

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/pytorch-2.8.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Why this project exists

Financial institutions routinely receive scanned documents (bank statements, ID cards,
invoices, proof of residence) that can be tampered with using image-editing software
or generative AI. **ForgeSight-Lite** is a small, focused portfolio project that
demonstrates how such tampering can be detected automatically using a two-stream
CNN that fuses **spatial (RGB)** and **frequency-domain (DCT)** features.

The project is intentionally scoped to be **understandable in one sitting** and
**reproducible on a single GPU**. It is not a state-of-the-art system — it is a
clean reference implementation of ideas from recent forensic CV literature.

## Key features

- **Two-stream architecture** — RGB stream (EfficientNet-B0) + DCT frequency stream, fused with a lightweight attention head.
- **End-to-end pipeline** — from raw image/PDF to forgery probability + tampering heatmap.
- **Production-ready packaging** — FastAPI endpoint, multi-stage Dockerfile, CPU-compatible inference.
- **Reproducible training** — PyTorch Lightning, single-command training on CASIA v2.
- **Evaluated properly** — AUC, F1, and pixel-level IoU reported on a held-out split.
- **Documented** — 2-page technical note in `paper/` explaining the design choices.

---

## Pipeline overview

### Training pipeline

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  CASIA v2    │────▶│  Preprocess  │────▶│  Augmentation │
  │  (raw)       │     │ (resize 256) │     │ (flip, JPEG)  │
  └──────────────┘     └──────────────┘     └───────┬──────┘
                                                    │
                ┌───────────────────────────────────┘
                │
                ▼
        ┌──────────────┐            ┌──────────────┐
        │  RGB stream  │            │  DCT stream  │
        │ EfficientNet │            │  (block DCT  │
        │     -B0      │            │  + small CNN)│
        └──────┬───────┘            └───────┬──────┘
               │                            │
               └────────────┬───────────────┘
                            ▼
                   ┌────────────────┐
                   │  Fusion head   │
                   │ (concat + MLP) │
                   └────────┬───────┘
                            ▼
                   ┌────────────────┐
                   │  BCE + Dice    │
                   │     loss       │
                   └────────┬───────┘
                            ▼
                   ┌────────────────┐
                   │   Checkpoint   │
                   │     (.ckpt)    │
                   └────────────────┘
```

### Inference pipeline (API)

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  HTTP POST   │────▶│ PDF? split   │────▶│  Resize 256  │
  │ /predict     │     │ into pages   │     │  Normalize   │
  │ (img or pdf) │     │ (pdf2image)  │     │              │
  └──────────────┘     └──────────────┘     └───────┬──────┘
                                                    │
                              ┌─────────────────────┤
                              ▼                     ▼
                      ┌──────────────┐      ┌──────────────┐
                      │  RGB stream  │      │  DCT stream  │
                      └──────┬───────┘      └───────┬──────┘
                             └──────────┬───────────┘
                                        ▼
                               ┌────────────────┐
                               │  Fusion head   │
                               └────────┬───────┘
                                        ▼
                               ┌────────────────┐
                               │  JSON response │
                               │  {             │
                               │   "score": …,  │
                               │   "heatmap":…  │
                               │  }             │
                               └────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- NVIDIA GPU with CUDA 12.8 (for training; inference works on CPU)
- `poppler` system package (for PDF input support)

  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - macOS: `brew install poppler`

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/forgesight-lite.git
cd forgesight-lite

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate          # Windows

# Install PyTorch first (pinned to the version this project was built with)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install the remaining dependencies
pip install -r requirements.txt
```

---

## Dataset

The project uses **CASIA v2**, a standard public dataset for image forgery detection
containing authentic, spliced, and copy-move images.

```bash
# Download and unpack CASIA v2 under ./data/casia2/
# (see data/README.md for the exact URL and expected folder layout)
python -m forgesight.data.download_casia --dest ./data/casia2
```

Expected layout after extraction:

```
data/casia2/
├── authentic/   # ~7k original images
├── tampered/    # ~5k forged images (splicing + copy-move)
└── masks/       # ground-truth tampering masks
```

---

## Quickstart

### 1. Train

```bash
python -m forgesight.train.train \
    --data-dir ./data/casia2 \
    --epochs 20 \
    --batch-size 32 \
    --output-dir ./runs/exp1
```

Training writes checkpoints, a CSV log, and a `best.ckpt` to `./runs/exp1/`.

### 2. Evaluate

```bash
python -m forgesight.eval.evaluate \
    --checkpoint ./runs/exp1/best.ckpt \
    --data-dir ./data/casia2
```

Prints AUC, F1, and pixel-level IoU on the held-out test split.

### 3. Serve the API

```bash
# Locally
uvicorn forgesight.api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t forgesight-lite .
docker run -p 8000:8000 -v $(pwd)/runs/exp1:/app/checkpoints forgesight-lite
```

Test the endpoint:

```bash
curl -X POST -F "file=@examples/fake_invoice.jpg" http://localhost:8000/predict
```

Example response:

```json
{
  "score": 0.87,
  "label": "forged",
  "heatmap_png_b64": "iVBORw0KGgoAAAANSUhEUgA..."
}
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project structure

```
forgesight-lite/
├── src/forgesight/
│   ├── data/                 # dataset loaders, DCT feature extraction, download script
│   ├── models/               # rgb_stream.py, freq_stream.py, fusion.py
│   ├── train/                # Lightning module and training entrypoint
│   ├── eval/                 # evaluation metrics and script
│   └── api/                  # FastAPI application
├── tests/                    # smoke tests
├── paper/                    # 2-page technical note (Markdown)
├── examples/                 # sample images for demo
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Results

Evaluated on a held-out 15% split of CASIA v2.

| Metric            | Value |
|-------------------|-------|
| Image-level AUC   | TBD   |
| Image-level F1    | TBD   |
| Pixel-level IoU   | TBD   |

*(Numbers will be filled in after the first full training run.)*

---

## Tech stack

| Layer         | Tool                                 |
|---------------|--------------------------------------|
| DL framework  | PyTorch 2.8 + PyTorch Lightning      |
| Backbones     | `timm` (EfficientNet-B0)             |
| Image I/O     | Pillow, OpenCV (headless), pdf2image |
| API           | FastAPI + Uvicorn                    |
| Packaging     | Docker (multi-stage)                 |
| Testing       | pytest                               |
| Lint / format | ruff                                 |

---

## References

The two-stream design is inspired by:

1. Kwon et al., *Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization* (CAT-Net), IJCV 2022.
2. Wu et al., *ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries With Anomalous Features*, CVPR 2019.
3. Guillaro et al., *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization*, CVPR 2023.

See `paper/technical_note.md` for a full discussion of the design choices.

---

## Future work

Things deliberately left out of this scope, but that would naturally extend the project:

- **Noise-residual stream** (SRM filters or Noiseprint) — the third forensic stream.
- **Vision Transformer backbone** (ViT / Swin) instead of EfficientNet.
- **Synthetic document generator** — programmatic alteration of clean templates with a diffusion inpainter, producing unlimited labeled training data.
- **AI-inpainting detection** — explicit benchmarks on Stable Diffusion / LaMa edits.
- **Cloud deployment** — GCP Cloud Run or AWS Lambda container.

---

## License

MIT. See [LICENSE](LICENSE).

## Author

Built as a portfolio project for an AI Computer Vision internship application.
Feedback and PRs welcome.

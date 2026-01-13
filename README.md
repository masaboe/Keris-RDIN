# Keris-RDIN
Keris-RDIN: end-to-end pipeline for keris image classification with mask-aware preprocessing, augmentation, and reproducible CSV evaluation (private dataset).
=======
# Keris-RDIN Repository

> **Note (Private Data):** The full dataset is **private** and is **not** included in this repository.  
> Please read `data/README.md` for the required local folder layout.

## 1) Pipeline

1. **Data acquisition** (private): multi-pose keris photos + multi-class labels  
2. **Pre-processing**
   - Background removal (**rembg/U^2-Net**) → alpha matte → mask cleanup → composite to **white background** (RGB PNG)
   - PCA-based vertical orientation + handle-down correction
   - Baseline normalization to **512×512**
   - YOLOv8 detector for *bilah* → crop → resize to **512×512**
   - QA checks (mask area / margin / white background consistency)
3. **Splitting**: train/val/test
4. **Augmentation** (mask-aware) + class balancing (oversample to majority)
5. **Modeling**: **Keris-RDIN** (InceptionResNetV2 + Residual Dilated Inception block + SE)
6. **Evaluation**: accuracy/precision/recall/F1/AUC + confusion matrix + CSV artifacts

This repository keeps the original notebooks under `notebooks/` and provides runnable CLI scripts under `scripts/` for reproducibility.

---

## 2) Repository layout

```
.
├── configs/                      # YAML configs
├── scripts/                      # Runnable pipeline stages (CLI)
├── src/keris/                    # Reusable python modules
├── notebooks/                    # Original experiment notebooks (reference)
├── data/
│   └── README.md                 # private data placement
├── THIRD_PARTY_NOTICES.md        # license notes for third-party tools
└── LICENSE
```

---

## 3) Installation

### Dependencies note
- Core training/evaluation uses TensorFlow/Keras + standard scientific Python packages.
- Stages require extra packages: `rembg` (Stage 01), `ultralytics` + `onnxruntime` (Stages 03a/03b).

### Option A — pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Option B — conda
```bash
conda env create -f environment.yml
conda activate keris-rdin
pip install -e .
```

---

## 4) Running the pipeline (CLI)

> All scripts support: `--config configs/<file>.yaml` and optional overrides via `--set key=value`.

### Stage 01 — Background removal (rembg)
```bash
python scripts/01_remove_bg.py --config configs/01_bg_remove.yaml \
  --set data.input_dir=data/raw_images \
  --set data.output_dir=data/no_background
```

### Stage 02 — PCA alignment + baseline normalization
```bash
python scripts/02_pca_align_baseline.py --config configs/02_pca_align_baseline.yaml \
  --set data.input_dir=data/no_background \
  --set data.output_dir=data/baselineNoBgWhite
```

### Stage 03a — Train YOLOv8
```bash
python scripts/03_train_yolo.py --config configs/03_train_yolo.yaml
```

### Stage 03b — YOLO crop bilah
```bash
python scripts/03b_crop_yolo.py --config configs/03b_crop_yolo.yaml \
  --set yolo.model_path=weights/yolo_bilah_best.onnx
```

### Stage 04 — Augmentation + balancing (NPY)
```bash
python scripts/04_augment_train.py --config configs/04_augment_train.yaml
```

### Stage 05 — Train Keris-RDIN (NPY)
```bash
python scripts/05_train_rdin.py --config configs/05_train_rdin.yaml
```

### Stage 06 — Evaluation (produces CSV tables)
```bash
python scripts/06_eval_rdin.py --config configs/06_eval_rdin.yaml
```

Outputs are written to `outputs/.../artifacts/`:
- `metrics_summary.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `predictions.csv`

These files are intended to reproduce the **main results tables** without releasing the full dataset.

---

## Reproducibility
- Random seed is configurable via YAML.
- Scripts log key environment versions (Python/CUDA and major libraries).
- Results are exported as CSV artifacts for manuscript tables and auditing.

---

## License and Third-Party Notices
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Citation
If you use this repository, please cite the associated manuscript (to be added).
<div align="center">

# 🛰️ MAVIC-T 2026: Multi-Modal Aerial View Image Translation

**4th Multi-modal Aerial View Imagery Challenge — Translation Track**  
**PBVS Workshop @ CVPR 2026 · Denver, CO, USA**

[![Rank](https://img.shields.io/badge/Leaderboard-🥈_2nd_Place-silver)](https://www.codabench.org/competitions/12566/)
[![Score](https://img.shields.io/badge/Combined_Score-0.51-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Team:** `pshradha` · **Author:** Shradhanjali Pradhan · **Affiliation:** Boston University

</div>

---

## 📋 Overview

This repository contains the complete solution for the [MAVIC-T 2026 challenge](https://www.codabench.org/competitions/12566/), which requires translating co-aligned aerial images between multiple sensor modalities. The challenge is part of the [Perception Beyond the Visible Spectrum (PBVS)](https://pbvs-workshop.github.io/) workshop at IEEE/CVF CVPR 2026.

Our approach uses a **hybrid strategy**: a deep learning model (U-Net GAN) for the data-rich SAR→EO task, and lightweight heuristic methods for the remaining data-scarce tasks.

| Task | Input → Output | Method | Parameters | Resolution |
|:----:|:--------------:|:------:|:----------:|:----------:|
| 1 | SAR → EO | **U-Net cGAN** | 54.4M + 2.8M | 256×256 |
| 2 | SAR → RGB | Histogram Matching | — | 256×256 |
| 3 | SAR → IR | Histogram Matching | — | 256×256 |
| 4 | RGB → IR | Luminance + Heuristic | — | 256×256 |

---

## 🏆 Results

### Final Leaderboard

| Rank | Participant | Combined ↓ | SAR→EO | SAR→RGB | RGB→IR | SAR→IR |
|:----:|:----------:|:----------:|:------:|:-------:|:------:|:------:|
| 1 | StagAI | 0.56 | 0.52 | 0.68 | 0.42 | 0.64 |
| **2** | **pshradha (ours)** | **0.51** | **0.51** | **0.56** | **0.42** | **0.55** |
| 3 | shadowchaser2 | 0.42 | 0.33 | 0.60 | 0.21 | 0.53 |
| 4 | bilisakura | 0.41 | 0.27 | 0.58 | 0.20 | 0.60 |
| 5 | wangzhiyu918 | 0.32 | 0.11 | 0.50 | 0.20 | 0.49 |

> **Scoring:** Per-task score = (2/π · arctan(FID) + LPIPS + L₁) / 3, averaged across 4 tasks. Lower is better.

---

## 🏗️ Architecture

### U-Net Generator (54.4M params)

```
SAR Input (1×256×256)
    │
    ▼ Encoder (8 blocks, strided 4×4 conv + InstanceNorm + LeakyReLU)
    ├─ down1: 1→64    (128×128)  [no norm]
    ├─ down2: 64→128   (64×64)
    ├─ down3: 128→256  (32×32)
    ├─ down4: 256→512  (16×16)
    ├─ down5: 512→512  (8×8)
    ├─ down6: 512→512  (4×4)
    ├─ down7: 512→512  (2×2)
    └─ down8: 512→512  (1×1)    [no norm, bottleneck]
    │
    ▼ Decoder (8 blocks, transposed conv + InstanceNorm + ReLU + skip concat)
    ├─ up1: 512→512   + skip(d7) → 1024  [dropout=0.5]
    ├─ up2: 1024→512  + skip(d6) → 1024  [dropout=0.5]
    ├─ up3: 1024→512  + skip(d5) → 1024  [dropout=0.5]
    ├─ up4: 1024→512  + skip(d4) → 1024
    ├─ up5: 1024→256  + skip(d3) → 512
    ├─ up6: 512→128   + skip(d2) → 256
    ├─ up7: 256→64    + skip(d1) → 128
    └─ final: 128→1 + Tanh
    │
    ▼
EO Output (1×256×256)
```

### PatchGAN Discriminator (2.8M params)

- Input: concat(SAR, EO) = 2 channels
- 4 conv layers → 31×31 prediction map
- 70×70 receptive field

### Training Loss

```
L_G = L_LSGAN + 100 · L_L1
```
- LSGAN (MSE-based) with label smoothing (real = 0.9) on discriminator only
- L1 reconstruction loss (λ = 100) for structural fidelity

---

## 📦 Installation

```bash
git clone https://github.com/shradhanjalipradhan/mavic-t-2nd.git
cd mavic-t-2nd
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU with ≥16 GB VRAM (Tesla T4 or better)

---

## 📂 Dataset

Download the MAVIC-T dataset from the [Codabench competition page](https://www.codabench.org/competitions/12566/).

<details>
<summary><b>Expected directory structure</b> (click to expand)</summary>

```
mavic-t-design-data/
├── design_data/design_data/
│   ├── SAR/train/                    # 68,151 PNG (256×256) — UNICORN
│   └── EO/train/                     # 68,151 PNG (256×256) — UNICORN
├── uc_davis_merged_chips_stacks/
│   └── uc_davis_merged_chips_stacks/
│       └── <location>/               # Multi-modal TIFF stacks (SAR, RGB, IR)
└── mavic_t_2025_test/
    ├── sar2eo/                       # 3,586 PNG test inputs
    ├── sar2rgb/                      # 60 TIFF test inputs
    ├── sar2ir/                       # 60 TIFF test inputs
    └── rgb2ir/                       # 60 TIFF test inputs
```

</details>

---

## 🚀 Usage

### 1. Train the U-Net GAN (SAR → EO)

```bash
python train.py --data_base /path/to/mavic-t-design-data --epochs 5 --batch_size 16
```

Training takes ~5–6 hours on a Tesla T4. Checkpoints are saved after every epoch to `weights/`.

### 2. Run Inference (all 4 tasks)

```bash
python inference.py \
    --data_base /path/to/mavic-t-design-data \
    --checkpoint weights/sar2eo_final.pth \
    --output_dir submission
```

This runs all 4 tasks in sequence (~15 minutes total):
1. **SAR→EO:** U-Net GAN forward pass (3,586 images)
2. **RGB→IR:** Grayscale conversion with water darkening (60 images)
3. **SAR→RGB:** Histogram matching using UC Davis reference (60 images)
4. **SAR→IR:** Histogram matching using UC Davis reference (60 images)

### 3. Package Submission

```bash
python package_submission.py --submission_dir submission
```

Creates `submission.zip` ready for upload to Codabench.

---

## 🖥️ Hardware & Software

| Component | Specification |
|:---------:|:------------:|
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| CPU | Intel Xeon (Kaggle runtime, 4 cores) |
| RAM | 13 GB |
| Language | Python 3.10 |
| Framework | PyTorch 2.0 |
| Libraries | NumPy, Pillow, rasterio, matplotlib |
| Training time | ~5–6 hours |
| Inference time | ~15 minutes (all 4 tasks) |

---

## 📁 Repository Structure

```
mavic-t-2nd/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── figures/                      # Sample outputs and visualizations
├── src/
│   ├── __init__.py
│   ├── model.py                 # U-Net Generator (54.4M) + PatchGAN (2.8M)
│   ├── dataset.py               # SAR-EO paired dataset with augmentation
│   └── heuristics.py            # Histogram matching + RGB→IR conversion
├── weights/                      # Model checkpoints (generated after training)
├── .gitignore
├── LICENSE                       # MIT License
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── train.py                      # Training script
├── inference.py                  # Inference for all 4 tasks
├── package_submission.py         # Create submission ZIP
└── mavic.ipynb                   # Original Kaggle notebook
```

---

## 📝 Technical Report

Our technical report submitted to the PBVS 2026 workshop is available in [`mavict_paper_files.zip`](mavict_paper_files.zip). It follows the CVPR 2026 author kit format and describes the complete approach, architecture, training details, and results.

---

## 📖 Citation

If you find this work useful, please cite the MAVIC-T challenge:

```bibtex
@inproceedings{mavict2024,
    title     = {Multi-modal Aerial View Image Challenge: Sensor Domain Translation},
    author    = {Low, Spencer and Nina, Oliver and Bowald, Dylan and 
                 Sappa, Angel D. and Inkawhich, Nathan and Bruns, Peter},
    booktitle = {CVPR Workshops},
    year      = {2024}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Made with 🛰️ for PBVS 2026 @ CVPR**

</div>

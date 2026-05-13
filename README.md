<div align="center">

# PG-M2TN

### Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

**Bridging Microscopic Polarization and Macroscopic Degradation:**  
*A Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics*

</div>

---

## Highlights

- 🔋 **Concurrent SOH + VDR Prediction** — First framework to simultaneously estimate State of Health (capacity fade) and Voltage Distortion Ratio (polarization indicator) in a unified model
- 🎭 **MAE Self-Supervised Learning** — Masked Autoencoder branch reconstructs fragmented IoT data, enabling robust predictions under 10%–90% data loss
- ⚖️ **Physics-Gated Dynamic Loss** — α-factor from IC curves adaptively weights SOH vs. VDR tasks, resolving the multi-task "seesaw effect"
- ⚡ **Edge-Deployable** — ~530K parameters, <1MB model size, O(n) complexity via BiLSTM backbone
- 📊 **5 Real-World Datasets** — Validated on CALCE, HUST, HNEI, CALB, and ISU-ILCC battery cycling data

## Architecture

```
Masked [V, I] → BiLSTM Encoder → ┬─ MAE Decoder  → Reconstruction (L_mae)
                                  ├─ SOH Head      → Capacity Fade  (L_soh)
                                  └─ VDR Head      → Polarization   (L_vdr)
                                        ↑
                           Physics-Gated α ──── IC Curve Analysis
```

### Key Design

| Component | Description |
|-----------|-------------|
| **Shared Encoder** | 2-layer BiLSTM (hidden=128), LayerNorm, ~530K params |
| **MAE Decoder** | MLP reconstructor for self-supervised regularization |
| **Attention Pooling** | 2-layer MLP attention (Tanh + Softmax) for time-step weighting |
| **SOH / VDR Heads** | Parallel MLP heads: Linear(256→64) → GELU → Linear(64→1) |
| **Physics Gating** | α from IC curves → w_SOH ∈ [0.50, 0.70], w_VDR ∈ [0.50, 0.70] |

## Results

### Comparison with Baselines (Pooled, 50% Mask Ratio)

| Model | SOH RMSE ↓ | SOH MAE ↓ | SOH R² ↑ | Params |
|-------|------------|-----------|----------|--------|
| 1D-CNN | 0.0423 | 0.0312 | 0.8847 | 18K |
| Standard GRU | 0.0298 | 0.0219 | 0.9428 | 165K |
| Standard LSTM | 0.0276 | 0.0198 | 0.9510 | 220K |
| Vanilla Transformer | 0.0301 | 0.0225 | 0.9416 | 789K |
| **PG-M2TN (Ours)** | **0.0182** | **0.0129** | **0.9787** | **530K** |

### Ablation Study

| Variant | SOH RMSE | VDR RMSE | Δ SOH |
|---------|----------|----------|-------|
| Full PG-M2TN | **0.0182** | **0.0245** | — |
| w/o MAE | 0.0231 | 0.0298 | +26.9% |
| w/o Dynamic Gating | 0.0195 | 0.0267 | +7.1% |
| w/o VDR (Single SOH) | 0.0218 | — | +19.8% |

## Installation

```bash
git clone https://github.com/shuhaochen618-svg/PG-M2TN.git
cd PG-M2TN

# Create environment
conda create -n pgm2tn python=3.11 -y
conda activate pgm2tn

# Install dependencies
pip install -r requirements.txt

# Install package (optional)
pip install -e .
```

## Data Preparation

Each cell should be stored as a `.pkl` file with the following structure:

```python
{
    'cell_id': 'CALCE_CS2_35',
    'nominal_capacity_in_Ah': 1.1,
    'cathode_material': 'LCO',       # optional
    'cycle_data': [
        {
            'voltage_in_V': np.array([...]),
            'current_in_A': np.array([...]),
            'charge_capacity_in_Ah': np.array([...]),
            'discharge_capacity_in_Ah': np.array([...]),
        },
        ...  # one dict per cycle
    ]
}
```

Organize under `./dataset/<DATASET_NAME>/`:

```
dataset/
├── CALCE/    *.pkl
├── HUST/     *.pkl
├── HNEI/     *.pkl
├── CALB/     *.pkl
└── ISU_ILCC/ *.pkl
```

Public sources: [CALCE](https://calce.umd.edu/battery-data) · [HNEI](https://www.hnei.hawaii.edu/)

## Quick Start

### Training (Single GPU)

```bash
python scripts/train.py \
    --data_root ./dataset \
    --datasets CALCE HUST HNEI \
    --epochs 150 \
    --batch_size 256
```

### Training (Multi-GPU DDP)

```bash
torchrun --nproc_per_node=8 scripts/train_ddp.py \
    --data_root ./dataset \
    --epochs 100 \
    --batch_size 128
```

### Ablation Modes

```bash
python scripts/train.py --ablation no_mae        # w/o MAE reconstruction
python scripts/train.py --ablation no_gating      # w/o physics gating (fixed 0.5/0.5)
python scripts/train.py --ablation no_vdr         # w/o VDR auxiliary task
python scripts/train.py --ablation single_task    # SOH only (no MAE, no VDR)
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/pgm2tn_none_best.pt \
    --data_root ./dataset
```

## Project Structure

```
PG-M2TN/
├── pg_m2tn/                      # Core Python package
│   ├── models/
│   │   ├── pg_m2tn.py            #   PG-M2TN architecture (BiLSTM + MAE + MTL)
│   │   ├── loss.py               #   Physics-gated dynamic loss
│   │   ├── physics_extractor.py  #   IC-curve α extraction (DDP-safe)
│   │   └── baselines.py          #   Baseline models (CNN, GRU, LSTM, Transformer)
│   ├── data/
│   │   ├── dataset_loader.py     #   Battery .pkl loader + stratified split
│   │   ├── masking_engine.py     #   Continuous block masking
│   │   └── download.py           #   Dataset download utility
│   └── utils/
│       ├── metrics.py            #   RMSE, MAE, MAPE, R²
│       ├── scheduler.py          #   Warmup + cosine annealing LR
│       └── plot_style.py         #   Nature-journal plotting style
│
├── scripts/
│   ├── train.py                  # Single-GPU training
│   ├── train_ddp.py              # Multi-GPU DDP training
│   ├── evaluate.py               # Checkpoint evaluation
│   └── run_ablation.sh           # One-click ablation runner
│
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Citation

```bibtex
@article{chen2026pgm2tn,
  title={Bridging Microscopic Polarization and Macroscopic Degradation:
         A Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics},
  author={Chen, Shuhao},
  journal={Applied Energy},
  year={2026}
}
```

## Contact

- **Shuhao Chen** — Zhejiang Sci-Tech University (ZSTU)
- Email: 2023333541008@mails.zstu.edu.cn

## License

This project is licensed under the [MIT License](LICENSE).

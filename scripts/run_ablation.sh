#!/bin/bash
# PG-M2TN Ablation Study Runner
# ================================
# Runs the full 2x2 factorial ablation study:
#   1. Full PG-M2TN (SOH + VDR + MAE + Dynamic Gating)
#   2. No-MAE       (SOH + VDR, no reconstruction)
#   3. No-VDR       (SOH + MAE, no auxiliary VDR task)
#   4. Single-Task  (SOH only, no MAE, no VDR)
#
# Usage:
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --data_root /path/to/dataset

DATA_ROOT=${1:-"./dataset"}
EPOCHS=150
BATCH=256
SAVE="./checkpoints"

echo "============================================="
echo "  PG-M2TN Ablation Study"
echo "  Data: $DATA_ROOT | Epochs: $EPOCHS"
echo "============================================="

for ABL in none no_mae no_vdr single_task; do
    echo ""
    echo ">>> Running ablation: $ABL"
    python scripts/train.py \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --ablation $ABL \
        --save_dir $SAVE
done

echo ""
echo "============================================="
echo "  All ablation runs complete!"
echo "  Results saved in $SAVE/"
echo "============================================="

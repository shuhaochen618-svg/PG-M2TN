"""
Baseline Per-Dataset Evaluation Script
========================================
Evaluates all baseline checkpoints (.pt) saved in exp0 to produce
per-dataset RMSE/MAE/MAPE/R² breakdowns for the 3D bar chart.

The baselines are SINGLE-TASK models (they only predict SOH, not VDR),
except for Hard-Sharing MTL which predicts both SOH and VDR.

PatchTST uses the server_deploy PatchTST_MAE architecture — handled
specially since it has a different forward() signature.

Usage (on GPU server):
  cd /path/to/PG-M2TN
  python scripts/eval_baselines_per_dataset.py \
      --ckpt_dir /path/to/web_result/result_2/results/exp0 \
      --data_root ./dataset \
      --output /path/to/web_result/result_2/results/exp0/baseline_per_dataset.json

After running, copy baseline_per_dataset.json back to your local machine at:
  E:\AI_codeing\web_result\result_2\results\exp0\baseline_per_dataset.json
"""

import os, sys, argparse, json, traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_m2tn.models.baselines import (
    CNN1D, StandardGRU, StandardLSTM, VanillaTransformer, HardSharingMTL
)
from pg_m2tn.models.pg_m2tn import PGM2TN
from pg_m2tn.data.dataset_loader import BatteryCycleDataset, split_by_cell
from pg_m2tn.data.masking_engine import MaskedBatteryDataset
from pg_m2tn.utils.metrics import compute_task_metrics


# ── Checkpoint → Model class mapping ────────────────────────────
# The .pt files in exp0 follow these naming conventions.
# Each returns a freshly constructed model (weights loaded separately).
BASELINE_REGISTRY = {
    '1D-CNN':               {'fn': lambda: CNN1D(input_dim=2, hidden_dim=64),
                             'mtl': False},
    'Standard_GRU':         {'fn': lambda: StandardGRU(input_dim=2, hidden_dim=128, num_layers=2),
                             'mtl': False},
    'Standard_LSTM':        {'fn': lambda: StandardLSTM(input_dim=2, hidden_dim=128, num_layers=2),
                             'mtl': False},
    'Vanilla_Transformer':  {'fn': lambda: VanillaTransformer(input_dim=2, d_model=128, n_heads=4, num_layers=4),
                             'mtl': False},
    'Hard-Sharing_MTL':     {'fn': lambda: HardSharingMTL(input_dim=2, hidden_dim=128, num_layers=2),
                             'mtl': True},
}


def get_dataset_prefix(cell_id, known=('CALCE', 'HUST', 'HNEI', 'CALB', 'ISU_ILCC')):
    """Infer dataset name from cell_id string."""
    uid = str(cell_id).upper().replace('-', '_')
    for ds in known:
        if uid.startswith(ds):
            return ds
    for ds in known:
        if ds in uid:
            return ds
    return 'Unknown'


def evaluate_model(model, dataloader, device, is_mtl=False):
    """
    Run inference and collect per-sample predictions.
    Returns dict with cell_ids, soh_pred, soh_true lists.
    """
    model.eval()
    soh_preds, soh_trues, cell_ids = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x_masked'].to(device)
            soh_true = batch['soh'].numpy()

            with autocast():
                out = model(x)

            if is_mtl:
                # Hard-Sharing MTL returns (soh_pred, vdr_pred)
                soh_pred = out[0].squeeze().cpu().numpy()
            else:
                soh_pred = out.squeeze().cpu().numpy()

            # Handle scalar output (batch_size=1 edge case)
            if soh_pred.ndim == 0:
                soh_pred = soh_pred.reshape(1)

            soh_preds.append(soh_pred)
            soh_trues.append(soh_true)
            cell_ids.extend(list(batch['cell_id']))

    return {
        'cell_ids': cell_ids,
        'soh_pred': np.concatenate(soh_preds).tolist(),
        'soh_true': np.concatenate(soh_trues).tolist(),
    }


def compute_per_ds_metrics(per_sample):
    """Group by dataset and compute metrics."""
    from collections import defaultdict
    groups = defaultdict(lambda: {'pred': [], 'true': []})

    for cid, sp, st in zip(per_sample['cell_ids'],
                           per_sample['soh_pred'],
                           per_sample['soh_true']):
        ds = get_dataset_prefix(cid)
        groups[ds]['pred'].append(sp)
        groups[ds]['true'].append(st)

    out = {}
    for ds, g in sorted(groups.items()):
        p = np.array(g['pred'])
        t = np.array(g['true'])
        m = compute_task_metrics(p - t, t)
        out[ds] = {
            'n': len(p),
            'rmse': m['rmse'],
            'mae': m['mae'],
            'mape': m['mape'],
            'r2': m['r2'],
        }
    return out


def try_load_patchtst(ckpt_dir, device):
    """
    Attempt to load PatchTST checkpoint.
    PatchTST uses server_deploy/models/patchtst_mae.py which has a
    different architecture. We try to auto-detect its structure from
    the checkpoint's state_dict keys.
    """
    pt_path = os.path.join(ckpt_dir, 'PatchTST.pt')
    if not os.path.exists(pt_path):
        return None

    print(f"\n  [PatchTST] Loading checkpoint: {pt_path}")
    ckpt = torch.load(pt_path, map_location=device)

    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any('encoder' in k for k in ckpt.keys()):
        sd = ckpt
    else:
        sd = ckpt

    # Print state_dict keys for debugging
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    print(f"  [PatchTST] State dict has {len(keys)} keys")
    if keys:
        print(f"  [PatchTST] First 5 keys: {keys[:5]}")

    # Try: maybe PatchTST was saved as a simple SOH-only Transformer variant
    # If it has 'head' keys, it might be a VanillaTransformer-like model
    # with different hyperparameters. We infer from state_dict.
    try:
        # Check if it matches a VanillaTransformer with larger d_model
        if any('transformer' in k for k in keys):
            # Infer d_model from embedding weight
            for k in keys:
                if 'embedding.weight' in k:
                    d_model = sd[k].shape[0]
                    break
            else:
                d_model = 256

            model = VanillaTransformer(input_dim=2, d_model=d_model, n_heads=8,
                                       num_layers=6, max_seq_len=1024).to(device)
            model.load_state_dict(sd)
            print(f"  [PatchTST] Loaded as VanillaTransformer variant (d_model={d_model})")
            return model
    except Exception:
        pass

    print(f"  [PatchTST] WARNING: Could not auto-load PatchTST. "
          f"Please check the model architecture manually.")
    print(f"  [PatchTST] Skipping PatchTST evaluation.")
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate baselines per-dataset')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory containing baseline .pt files (exp0/)')
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help='Root directory of processed datasets')
    parser.add_argument('--datasets', nargs='+',
                        default=['CALCE', 'HUST', 'HNEI', 'CALB', 'ISU_ILCC'])
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Fixed mask ratio for test set (same as main experiment)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path. Default: <ckpt_dir>/baseline_per_dataset.json')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.ckpt_dir, 'baseline_per_dataset.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Data root: {args.data_root}")

    # ── Load dataset & create test split ─────────────────────────
    print("\nLoading dataset...")
    base_ds = BatteryCycleDataset(args.data_root, args.datasets, min_cycles=50)
    _, _, test_idx = split_by_cell(base_ds, seed=args.seed)
    test_ds = MaskedBatteryDataset(Subset(base_ds, test_idx),
                                    fixed_mask_ratio=args.mask_ratio, seed=42)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Test set: {len(test_ds)} samples")

    # ── Evaluate each registered baseline ────────────────────────
    all_results = {}

    for ckpt_name, cfg in BASELINE_REGISTRY.items():
        pt_path = os.path.join(args.ckpt_dir, f'{ckpt_name}.pt')
        if not os.path.exists(pt_path):
            print(f"\n[SKIP] {pt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {ckpt_name}")
        print(f"  Checkpoint: {pt_path}")

        try:
            model = cfg['fn']().to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

            ckpt = torch.load(pt_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)

            per_sample = evaluate_model(model, test_dl, device, is_mtl=cfg['mtl'])

            # Pooled metrics
            sp = np.array(per_sample['soh_pred'])
            st = np.array(per_sample['soh_true'])
            pooled = compute_task_metrics(sp - st, st)
            print(f"  Pooled SOH — RMSE: {pooled['rmse']:.4f} | MAE: {pooled['mae']:.4f} | "
                  f"MAPE: {pooled['mape']:.2f}% | R²: {pooled['r2']:.4f}")

            # Per-dataset metrics
            per_ds = compute_per_ds_metrics(per_sample)
            for ds, m in per_ds.items():
                print(f"    {ds:<12} N={m['n']:>5}  RMSE={m['rmse']:.4f}  "
                      f"MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%  R²={m['r2']:.4f}")

            display_name = ckpt_name.replace('_', ' ')
            all_results[display_name] = {
                'params': n_params,
                'pooled': pooled,
                'per_dataset': per_ds,
            }
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {ckpt_name}: {e}")
            traceback.print_exc()
            continue

    # ── Try PatchTST (special handling) ──────────────────────────
    ptst_model = try_load_patchtst(args.ckpt_dir, device)
    if ptst_model is not None:
        try:
            per_sample = evaluate_model(ptst_model, test_dl, device, is_mtl=False)
            sp = np.array(per_sample['soh_pred'])
            st = np.array(per_sample['soh_true'])
            pooled = compute_task_metrics(sp - st, st)
            per_ds = compute_per_ds_metrics(per_sample)
            n_params = sum(p.numel() for p in ptst_model.parameters())
            all_results['PatchTST'] = {
                'params': n_params,
                'pooled': pooled,
                'per_dataset': per_ds,
            }
            print(f"  PatchTST Pooled SOH — RMSE: {pooled['rmse']:.4f}")
        except Exception as e:
            print(f"  [ERROR] PatchTST evaluation failed: {e}")
            traceback.print_exc()

    # ── Also add PG-M2TN from existing baseline_comparison.json ──
    bc_path = os.path.join(args.ckpt_dir, 'baseline_comparison.json')
    if os.path.exists(bc_path):
        with open(bc_path, 'r') as f:
            bc = json.load(f)
        if 'pgm2tn_per_dataset' in bc:
            pgm2tn_ds = {}
            for ds, m in bc['pgm2tn_per_dataset'].items():
                if ds.startswith('_'):
                    continue
                pgm2tn_ds[ds] = {
                    'n': m['n'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'mape': m['mape'],
                    'r2': m['r2'],
                }
            all_results['PG-M2TN (Ours)'] = {
                'params': bc['pooled_results']['PG-M2TN (Ours)'].get('params', 630149),
                'pooled': bc['pooled_results']['PG-M2TN (Ours)']['metrics'],
                'per_dataset': pgm2tn_ds,
            }
            print(f"\n  Added PG-M2TN (Ours) from baseline_comparison.json")

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS SAVED: {args.output}")
    print(f"  Models evaluated: {list(all_results.keys())}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

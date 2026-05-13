"""
PG-M2TN Evaluation Script
===========================
Evaluate a trained PG-M2TN checkpoint on test data.

Usage:
  python scripts/evaluate.py --checkpoint ./checkpoints/pgm2tn_none_best.pt
"""
import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters
from pg_m2tn.models.loss import PhysicsGatedLoss
from pg_m2tn.models.physics_extractor import PhysicsExtractor
from pg_m2tn.data.dataset_loader import BatteryCycleDataset, split_by_cell
from pg_m2tn.data.masking_engine import MaskedBatteryDataset
from pg_m2tn.utils.metrics import compute_task_metrics, compute_per_dataset_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, default='./dataset')
    p.add_argument('--datasets', nargs='+', default=['CALCE','HUST','HNEI','CALB','ISU_ILCC'])
    p.add_argument('--mask_ratio', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=256)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get('args', {})

    # Reconstruct model
    abl = ckpt_args.get('ablation', 'none')
    use_mae = abl not in ('no_mae','single_task')
    model = PGM2TN(2, ckpt_args.get('hidden_dim',128), ckpt_args.get('num_layers',2),
                   enable_mae=use_mae).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    print(f"Parameters: {count_parameters(model):,}")

    # Data
    ds = BatteryCycleDataset(args.data_root, args.datasets, min_cycles=50)
    _, _, te_i = split_by_cell(ds, seed=ckpt_args.get('seed',42))
    te_ds = MaskedBatteryDataset(Subset(ds, te_i), fixed_mask_ratio=args.mask_ratio, seed=42)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    phys = PhysicsExtractor()
    sp_all, st_all, vp_all, vt_all, cids = [],[],[],[],[]

    with torch.no_grad():
        for batch in te_dl:
            xm = batch['x_masked'].to(device)
            with autocast():
                _, sp, vp = model(xm)
            sp_all.append(sp.squeeze().cpu().numpy())
            st_all.append(batch['soh'].numpy())
            vp_all.append(vp.squeeze().cpu().numpy())
            vt_all.append(batch['vdr'].numpy())
            cids.extend(list(batch['cell_id']))

    sp_a, st_a = np.concatenate(sp_all), np.concatenate(st_all)
    vp_a, vt_a = np.concatenate(vp_all), np.concatenate(vt_all)
    sm = compute_task_metrics(sp_a-st_a, st_a)
    vm = compute_task_metrics(vp_a-vt_a, vt_a)

    print(f"\n{'='*60}")
    print(f"  SOH — RMSE: {sm['rmse']:.4f} | MAE: {sm['mae']:.4f} | "
          f"MAPE: {sm['mape']:.2f}% | R²: {sm['r2']:.4f}")
    print(f"  VDR — RMSE: {vm['rmse']:.4f} | MAE: {vm['mae']:.4f} | R²: {vm['r2']:.4f}")

    pd = compute_per_dataset_metrics({
        'cell_ids':cids, 'soh_pred':sp_a.tolist(), 'soh_true':st_a.tolist(),
        'vdr_pred':vp_a.tolist(), 'vdr_true':vt_a.tolist()})
    print(f"\n  Per-Dataset Breakdown:")
    for d,m in pd.items():
        if d.startswith('_'): continue
        print(f"    {d:<12} N={m['n_samples']:>5} SOH_RMSE={m['soh']['rmse']:.4f} "
              f"VDR_RMSE={m['vdr']['rmse']:.4f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

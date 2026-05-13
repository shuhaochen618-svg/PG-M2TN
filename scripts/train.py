"""
PG-M2TN Single-GPU Training Script
=====================================
Usage:
  python scripts/train.py --data_root ./dataset --epochs 100
  python scripts/train.py --ablation no_mae
"""
import os, sys, time, datetime, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters
from pg_m2tn.models.loss import PhysicsGatedLoss
from pg_m2tn.models.physics_extractor import PhysicsExtractor
from pg_m2tn.data.dataset_loader import BatteryCycleDataset, split_by_cell
from pg_m2tn.data.masking_engine import MaskedBatteryDataset
from pg_m2tn.utils.metrics import compute_task_metrics
from pg_m2tn.utils.scheduler import WarmupCosineScheduler

def get_args():
    p = argparse.ArgumentParser(description='PG-M2TN Training')
    p.add_argument('--data_root', type=str, default='./dataset')
    p.add_argument('--datasets', nargs='+', default=['CALCE','HUST','HNEI','CALB','ISU_ILCC'])
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--min_cycles', type=int, default=50)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--lambda_mae', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--ablation', type=str, default='none',
                   choices=['none','no_mae','no_gating','no_vdr','single_task'])
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, scaler, phys, device, use_mae, use_gating, use_vdr):
    model.train()
    total, accum, nb = 0., {'mae':0,'soh':0,'vdr':0}, 0
    for batch in loader:
        xm = batch['x_masked'].to(device); xf = batch['x_full'].to(device)
        soh = batch['soh'].to(device); vdr = batch['vdr'].to(device)
        alpha = phys.batch_extract(batch['V_raw'], batch['Q_raw'],
            cell_ids=batch['cell_id'], cycle_indices=batch['cycle_idx'],
            soh_batch=batch['soh']).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            xr, sp, vp = model(xm)
            loss, ld = criterion(xf, xr, soh, sp, vdr, vp, alpha,
                use_dynamic_gating=use_gating, use_mae=use_mae, use_vdr=use_vdr)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer); scaler.update()
        total += ld['total']
        for k in accum: accum[k] += ld[k]
        nb += 1
    avg = {k: v/max(nb,1) for k,v in accum.items()}
    avg['total'] = total/max(nb,1)
    return avg

@torch.no_grad()
def evaluate(model, loader, criterion, phys, device, use_mae, use_gating, use_vdr):
    model.eval()
    total, accum, nb = 0., {'mae':0,'soh':0,'vdr':0}, 0
    sp_all, st_all, vp_all, vt_all, mr_all = [],[],[],[],[]
    for batch in loader:
        xm = batch['x_masked'].to(device); xf = batch['x_full'].to(device)
        soh = batch['soh'].to(device); vdr = batch['vdr'].to(device)
        alpha = phys.batch_extract(batch['V_raw'], batch['Q_raw'],
            cell_ids=batch['cell_id'], cycle_indices=batch['cycle_idx'],
            soh_batch=batch['soh'], inference_mode=True).to(device)
        with autocast():
            xr, sp, vp = model(xm)
            loss, ld = criterion(xf, xr, soh, sp, vdr, vp, alpha,
                use_dynamic_gating=use_gating, use_mae=use_mae, use_vdr=use_vdr)
        total += ld['total']
        for k in accum: accum[k] += ld[k]
        sp_all.append(sp.squeeze().cpu().numpy()); st_all.append(soh.cpu().numpy())
        vp_all.append(vp.squeeze().cpu().numpy()); vt_all.append(vdr.cpu().numpy())
        if use_mae and xr is not None:
            mr_all.append(((xr-xf)**2).mean(dim=(1,2)).sqrt().cpu().numpy())
        nb += 1
    sm = compute_task_metrics(np.concatenate(sp_all)-np.concatenate(st_all), np.concatenate(st_all))
    vm = compute_task_metrics(np.concatenate(vp_all)-np.concatenate(vt_all), np.concatenate(vt_all))
    mr = float(np.mean(np.concatenate(mr_all))) if mr_all else 0.
    al = {k: v/max(nb,1) for k,v in accum.items()}
    return {'loss':total/max(nb,1), 'soh_rmse':sm['rmse'], 'soh_mae':sm['mae'],
            'soh_mape':sm['mape'], 'soh_r2':sm['r2'], 'vdr_rmse':vm['rmse'],
            'vdr_mae':vm['mae'], 'vdr_r2':vm['r2'], 'mae_recon_rmse':mr,
            'loss_mae':al['mae'], 'loss_soh':al['soh'], 'loss_vdr':al['vdr']}

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_mae = args.ablation not in ('no_mae','single_task')
    use_gating = (args.ablation == 'none')
    use_vdr = args.ablation not in ('no_vdr','single_task')

    print("="*70); print("  PG-M2TN Training Pipeline"); print("="*70)
    print(f"  Device: {device} | Ablation: {args.ablation} | Epochs: {args.epochs}")

    # Data
    ds = BatteryCycleDataset(args.data_root, args.datasets, args.seq_len, args.min_cycles)
    tr_i, va_i, te_i = split_by_cell(ds, seed=args.seed)
    tr_dl = DataLoader(MaskedBatteryDataset(Subset(ds,tr_i)), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(MaskedBatteryDataset(Subset(ds,va_i)), batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(MaskedBatteryDataset(Subset(ds,te_i), fixed_mask_ratio=0.5),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = PGM2TN(2, args.hidden_dim, args.num_layers, args.dropout, use_mae).to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    phys = PhysicsExtractor()
    crit = PhysicsGatedLoss(lambda_mae=args.lambda_mae)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineScheduler(opt, args.warmup_epochs, args.epochs, args.lr)
    scaler = GradScaler()
    os.makedirs(args.save_dir, exist_ok=True)
    cn = f"pgm2tn_{args.ablation}"

    # Train
    best_vl, best_sr, pat, hist = float('inf'), float('inf'), 0, []
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        te = time.time(); sched.step(ep-1)
        tm = train_one_epoch(model, tr_dl, crit, opt, scaler, phys, device, use_mae, use_gating, use_vdr)
        vm = evaluate(model, va_dl, crit, phys, device, use_mae, use_gating, use_vdr)
        el = time.time()-te; ib = vm['loss']<best_vl
        mk = " *BEST*" if ib else ""
        if vm['soh_rmse']<best_sr: best_sr=vm['soh_rmse']
        print(f" {ep:>3d}/{args.epochs:<3d} | TrL:{tm['total']:.4f} | VaL:{vm['loss']:.4f} | "
              f"SOH:{vm['soh_rmse']:.4f} | VDR:{vm['vdr_rmse']:.4f} | R²:{vm['soh_r2']:.3f} | "
              f"{el:.1f}s{mk}")
        hist.append({'epoch':ep,'train_loss':tm['total'],'val_loss':vm['loss'],
                     'val_soh_rmse':vm['soh_rmse'],'val_soh_r2':vm['soh_r2']})
        if ib:
            best_vl=vm['loss']; pat=0
            torch.save({'epoch':ep,'model_state_dict':model.state_dict(),
                        'val_loss':best_vl,'args':vars(args)},
                       os.path.join(args.save_dir, f'{cn}_best.pt'))
        else:
            pat+=1
            if pat>=args.patience: print(f"  >> Early stopping at epoch {ep}"); break
    tt = time.time()-t0

    # Test
    ckpt = torch.load(os.path.join(args.save_dir, f'{cn}_best.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    tm = evaluate(model, te_dl, crit, phys, device, use_mae, use_gating, use_vdr)
    print(f"\n{'='*70}\n  TEST: SOH RMSE={tm['soh_rmse']:.4f} MAE={tm['soh_mae']:.4f} "
          f"R²={tm['soh_r2']:.4f} | VDR RMSE={tm['vdr_rmse']:.4f}\n{'='*70}")
    res = {'ablation':args.ablation, **{f'test_{k}':v for k,v in tm.items()},
           'params':count_parameters(model), 'time_s':tt, 'history':hist, 'args':vars(args)}
    rp = os.path.join(args.save_dir, f'{cn}_results.json')
    with open(rp,'w') as f: json.dump(res, f, indent=2)
    print(f"  Saved: {rp}")

if __name__ == '__main__':
    main()

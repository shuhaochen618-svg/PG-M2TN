"""
PG-M2TN DDP Training Pipeline
================================
Multi-GPU Distributed Data Parallel (DDP) training.

Usage:
  # Full training (8 GPUs)
  torchrun --nproc_per_node=8 train_ddp.py --data_root ./dataset --epochs 100

  # Ablation: No MAE
  torchrun --nproc_per_node=8 train_ddp.py --ablation no_mae

  # Ablation: No Gating
  torchrun --nproc_per_node=8 train_ddp.py --ablation no_gating
"""

import os
import sys
import time
import datetime
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_m2tn.data.dataset_loader import BatteryCycleDataset, split_by_cell
from pg_m2tn.data.masking_engine import MaskedBatteryDataset
from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters
from pg_m2tn.models.physics_extractor import PhysicsExtractor
from pg_m2tn.models.loss import PhysicsGatedLoss


# ====================================================================
# Configuration
# ====================================================================
def get_args():
    parser = argparse.ArgumentParser(description='PG-M2TN DDP Training')

    # Data
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--datasets', nargs='+',
                        default=['CALCE', 'HUST', 'HNEI', 'CALB', 'ISU_ILCC'])
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--min_cycles', type=int, default=50)

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Per-GPU batch size. Total = batch_size * num_gpus')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Base LR per GPU. Effective LR = lr * world_size.')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lambda_mae', type=float, default=0.1,
                        help='Weight for MAE reconstruction loss. Set to 0.1 to act as a soft regularizer.')
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Linear LR warmup epochs. 10 epochs is safer for multi-task training.')

    # Ablation
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_mae', 'no_gating', 'no_vdr', 'single_task'])

    # System
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


# ====================================================================
# DDP Utilities
# ====================================================================
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def print_rank0(*args, **kwargs):
    """Only print on rank 0."""
    if is_main_process():
        print(*args, **kwargs, flush=True)


# ====================================================================
# Learning Rate Scheduler with Warmup
# ====================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ====================================================================
# Training Loop
# ====================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler,
                    physics_extractor, device, use_mae, use_gating, use_vdr=True):
    model.train()
    total_loss = 0.0
    loss_accum = {'mae': 0, 'soh': 0, 'vdr': 0}
    num_batches = 0

    for batch in dataloader:
        x_masked = batch['x_masked'].to(device, non_blocking=True)
        x_full = batch['x_full'].to(device, non_blocking=True)
        soh = batch['soh'].to(device, non_blocking=True)
        vdr = batch['vdr'].to(device, non_blocking=True)
        V_raw = batch['V_raw']
        Q_raw = batch['Q_raw']
        cell_ids = batch['cell_id']
        cycle_indices = batch['cycle_idx']

        alpha = physics_extractor.batch_extract(
            V_raw, Q_raw, cell_ids=cell_ids, cycle_indices=cycle_indices,
            soh_batch=batch['soh']
        ).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            x_recon, soh_pred, vdr_pred = model(x_masked)
            loss, loss_dict = criterion(
                x_full, x_recon,
                soh, soh_pred, vdr, vdr_pred,
                alpha,
                use_dynamic_gating=use_gating,
                use_mae=use_mae,
                use_vdr=use_vdr,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict['total']
        for k in ['mae', 'soh', 'vdr']:
            loss_accum[k] += loss_dict[k]
        num_batches += 1

    avg = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}
    avg['total'] = total_loss / max(num_batches, 1)
    return avg


def _compute_task_metrics(errors, trues):
    """Compute RMSE, MAE, MAPE (raw + filtered), R2 for a single task.

    mape_filtered: excludes samples where |true| < 0.5 (End-of-Life cells
    with SOH → 0 cause MAPE denominator explosion). This is the standard
    practical MAPE used in Applied Energy battery prognostics literature.
    """
    errors = np.array(errors, dtype=np.float64)
    trues  = np.array(trues,  dtype=np.float64)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae  = float(np.mean(np.abs(errors)))
    safe = np.abs(trues) > 1e-6
    mape = float(np.mean(np.abs(errors[safe] / trues[safe]))) * 100 if safe.sum() > 0 else 0.0
    # Practical MAPE: exclude End-of-Life samples (|true| < 0.5)
    practical = np.abs(trues) >= 0.5
    safe_p = practical & safe
    mape_filtered = float(np.mean(np.abs(errors[safe_p] / trues[safe_p]))) * 100 if safe_p.sum() > 0 else 0.0
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot >= 1e-4 else 0.0
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'mape_filtered': mape_filtered, 'r2': r2}


def _get_dataset_prefix(cell_id: str, known: tuple = ('CALCE','HUST','HNEI','CALB','ISU_ILCC')) -> str:
    """
    Infer dataset name from cell_id string.

    Handles naming variants:
      'HUST_1-2'                          → 'HUST'
      'ISU-ILCC_G10C4'                    → 'ISU_ILCC'  (hyphen → underscore)
      'CALCE_CS2_35'                      → 'CALCE'
    Falls back to 'Unknown' if no prefix matches.
    """
    # Normalize: upper-case + replace hyphens with underscores
    uid = str(cell_id).upper().replace('-', '_')
    for ds in known:
        if uid.startswith(ds):
            return ds
    # Secondary: substring match (covers edge-case naming)
    for ds in known:
        if ds in uid:
            return ds
    return 'Unknown'


@torch.no_grad()
def evaluate(model, dataloader, criterion, physics_extractor, device,
             use_mae, use_gating, use_vdr=True, return_per_sample=False):
    """
    Evaluate model on dataloader.

    Args:
        return_per_sample: if True, also return lists of per-sample
            (cell_id, soh_pred, soh_true, vdr_pred, vdr_true) for
            downstream per-dataset breakdown.
    """
    model.eval()
    total_loss = 0.0
    loss_accum = {'mae': 0, 'soh': 0, 'vdr': 0}
    soh_preds, soh_trues = [], []
    vdr_preds, vdr_trues = [], []
    mae_recon_preds = []
    cell_ids_all = []   # for per-dataset breakdown
    cycle_idxs_all = [] # for trajectory plots
    num_batches = 0

    for batch in dataloader:
        x_masked = batch['x_masked'].to(device, non_blocking=True)
        x_full   = batch['x_full'].to(device, non_blocking=True)
        soh = batch['soh'].to(device, non_blocking=True)
        vdr = batch['vdr'].to(device, non_blocking=True)
        V_raw        = batch['V_raw']
        Q_raw        = batch['Q_raw']
        cell_ids     = batch['cell_id']
        cycle_indices= batch['cycle_idx']

        alpha = physics_extractor.batch_extract(
            V_raw, Q_raw, cell_ids=cell_ids, cycle_indices=cycle_indices,
            soh_batch=batch['soh'], inference_mode=True
        ).to(device, non_blocking=True)

        with autocast():
            x_recon, soh_pred, vdr_pred = model(x_masked)
            loss, loss_dict = criterion(
                x_full, x_recon,
                soh, soh_pred, vdr, vdr_pred,
                alpha,
                use_dynamic_gating=use_gating,
                use_mae=use_mae,
                use_vdr=use_vdr,
            )

        total_loss += loss_dict['total']
        for k in ['mae', 'soh', 'vdr']:
            loss_accum[k] += loss_dict[k]

        soh_preds.append(soh_pred.squeeze().cpu().numpy())
        soh_trues.append(soh.cpu().numpy())
        vdr_preds.append(vdr_pred.squeeze().cpu().numpy())
        vdr_trues.append(vdr.cpu().numpy())
        cell_ids_all.extend(list(cell_ids))  # str list
        cycle_idxs_all.extend([int(c) for c in cycle_indices])  # int list

        if use_mae and x_recon is not None:
            recon_rmse = ((x_recon - x_full) ** 2).mean(dim=(1, 2)).sqrt()
            mae_recon_preds.append(recon_rmse.cpu().numpy())

        num_batches += 1

    soh_p = np.concatenate(soh_preds)
    soh_t = np.concatenate(soh_trues)
    vdr_p = np.concatenate(vdr_preds)
    vdr_t = np.concatenate(vdr_trues)

    soh_m = _compute_task_metrics(soh_p - soh_t, soh_t)
    vdr_m = _compute_task_metrics(vdr_p - vdr_t, vdr_t)

    if mae_recon_preds:
        recon_vals = np.concatenate(mae_recon_preds)
        mae_m = {'rmse': float(np.mean(recon_vals)), 'mae': float(np.mean(recon_vals)),
                 'mape': 0.0, 'r2': 0.0}
    else:
        mae_m = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0}

    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    result = {
        'loss': total_loss / max(num_batches, 1),
        'soh_rmse': soh_m['rmse'], 'soh_mae': soh_m['mae'],
        'soh_mape': soh_m['mape'], 'soh_mape_filtered': soh_m['mape_filtered'],
        'soh_r2': soh_m['r2'],
        'vdr_rmse': vdr_m['rmse'], 'vdr_mae': vdr_m['mae'],
        'vdr_mape': vdr_m['mape'], 'vdr_mape_filtered': vdr_m['mape_filtered'],
        'vdr_r2': vdr_m['r2'],
        'mae_recon_rmse': mae_m['rmse'], 'mae_recon_mae': mae_m['mae'],
        'loss_mae': avg_losses['mae'], 'loss_soh': avg_losses['soh'],
        'loss_vdr': avg_losses['vdr'],
    }

    if return_per_sample:
        result['_per_sample'] = {
            'cell_ids':   cell_ids_all,
            'cycle_idxs': cycle_idxs_all,
            'soh_pred': soh_p.tolist(),
            'soh_true': soh_t.tolist(),
            'vdr_pred': vdr_p.tolist(),
            'vdr_true': vdr_t.tolist(),
        }
    return result


def compute_per_dataset_metrics(per_sample: dict) -> dict:
    """
    Group per-sample predictions by dataset prefix and compute metrics.

    Returns dict:
      {
        'CALCE': {'n_samples': int, 'soh': {...}, 'vdr': {...}},
        'HUST':  {...},
        ...
        '_macro_avg': {'soh': {...}, 'vdr': {...}}   # unweighted avg across datasets
      }
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {'soh_p': [], 'soh_t': [], 'vdr_p': [], 'vdr_t': []})

    for cid, sp, st, vp, vt in zip(
            per_sample['cell_ids'],
            per_sample['soh_pred'], per_sample['soh_true'],
            per_sample['vdr_pred'], per_sample['vdr_true']):
        ds = _get_dataset_prefix(cid)
        groups[ds]['soh_p'].append(sp)
        groups[ds]['soh_t'].append(st)
        groups[ds]['vdr_p'].append(vp)
        groups[ds]['vdr_t'].append(vt)

    out = {}
    soh_rmses, vdr_rmses = [], []
    for ds, g in sorted(groups.items()):
        soh_p = np.array(g['soh_p']); soh_t = np.array(g['soh_t'])
        vdr_p = np.array(g['vdr_p']); vdr_t = np.array(g['vdr_t'])
        soh_m = _compute_task_metrics(soh_p - soh_t, soh_t)
        vdr_m = _compute_task_metrics(vdr_p - vdr_t, vdr_t)
        out[ds] = {'n_samples': len(soh_p), 'soh': soh_m, 'vdr': vdr_m}
        soh_rmses.append(soh_m['rmse'])
        vdr_rmses.append(vdr_m['rmse'])

    # Macro-average (unweighted across datasets — treats each dataset equally)
    if soh_rmses:
        out['_macro_avg'] = {
            'soh': {'rmse': float(np.mean(soh_rmses))},
            'vdr': {'rmse': float(np.mean(vdr_rmses))},
            'note': 'Unweighted mean across datasets (each dataset counts equally)'
        }
    return out


# ====================================================================
# Main
# ====================================================================
def main():
    args = get_args()
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    # Seed
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # Linear scaling rule for LR
    effective_lr = args.lr * world_size
    effective_batch = args.batch_size * world_size

    # Ablation flags
    # New design: 'no_gating' is our FULL model (SOH + VDR + MAE, fixed 0.50/0.50)
    # All ablation variants also use fixed weights (no dynamic gating)
    use_mae = args.ablation not in ('no_mae', 'single_task')
    use_gating = (args.ablation == 'none')  # only 'none' uses dynamic gating
    use_vdr = args.ablation not in ('no_vdr', 'single_task')

    print_rank0()
    print_rank0("=" * 70)
    print_rank0("  PG-M2TN DDP Training  |  Multi-GPU")
    print_rank0("=" * 70)
    print_rank0(f"  World Size     : {world_size} GPUs")
    print_rank0(f"  Per-GPU Batch  : {args.batch_size}")
    print_rank0(f"  Total Batch    : {effective_batch}")
    print_rank0(f"  Base LR        : {args.lr} -> Effective LR: {effective_lr}")
    print_rank0(f"  Ablation       : {args.ablation}")
    print_rank0(f"  Datasets       : {args.datasets}")
    print_rank0(f"  Started at     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_rank0("=" * 70)

    # --- Data (load on all ranks) ---
    print_rank0("\n[1/4] Loading data...")
    t0 = time.time()
    base_dataset = BatteryCycleDataset(
        data_root=args.data_root, datasets=args.datasets,
        seq_len=args.seq_len, min_cycles=args.min_cycles,
    )
    train_idx, val_idx, test_idx = split_by_cell(base_dataset, seed=args.seed)
    print_rank0(f"  Data loaded in {time.time()-t0:.1f}s")

    train_ds = MaskedBatteryDataset(Subset(base_dataset, train_idx))
    # Validation: random mask ratio [0.1, 0.9] to test generalization across all NC levels
    val_ds   = MaskedBatteryDataset(Subset(base_dataset, val_idx))
    # Test: fixed 0.5 (moderate loss rate — most representative for final reporting)
    # Exp1 separately tests 0.1~0.9 for the robustness figure.
    test_ds  = MaskedBatteryDataset(Subset(base_dataset, test_idx), fixed_mask_ratio=0.5)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print_rank0(f"  Train: {len(train_ds)} samples ({len(train_loader)} batches/GPU/epoch)")
    print_rank0(f"  Val:   {len(val_ds)} samples")
    print_rank0(f"  Test:  {len(test_ds)} samples")

    # --- Model ---
    print_rank0("\n[2/4] Building DDP model...")
    model = PGM2TN(input_dim=2, hidden_dim=args.hidden_dim,
                   num_layers=args.num_layers, dropout=args.dropout,
                   enable_mae=use_mae).to(device)
    # When ablation disables VDR loss (no_vdr, single_task), the VDR head
    # parameters receive no gradient. DDP requires find_unused_parameters=True.
    has_unused = not use_vdr  # VDR head params unused when use_vdr=False
    model = DDP(model, device_ids=[local_rank],
                find_unused_parameters=has_unused)
    print_rank0(f"  Parameters: {count_parameters(model):,}")

    physics_extractor = PhysicsExtractor()
    criterion = PhysicsGatedLoss(lambda_mae=args.lambda_mae)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs,
                                      args.epochs, effective_lr)
    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = f"pgm2tn_{args.ablation}"

    # --- Training ---
    print_rank0("\n[3/4] Training (Multi-Task: SOH + VDR + MAE)...")
    print_rank0("-" * 110)
    print_rank0(f"{'Epoch':>7s} | {'Train Loss':>10s} | {'Val Loss':>10s} | "
                f"{'SOH RMSE':>9s} | {'VDR RMSE':>9s} | {'MAE Recon':>9s} | "
                f"{'R2':>6s} | {'Time':>6s} | {'ETA':>10s}")
    print_rank0("-" * 110)

    best_val_loss = float('inf')
    best_soh_rmse = float('inf')
    patience_counter = 0
    history = []
    epoch_times = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_sampler.set_epoch(epoch)  # CRITICAL for proper DDP shuffling

        current_lr = scheduler.step(epoch - 1)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            physics_extractor, device, use_mae, use_gating, use_vdr,
        )

        # Only rank 0 does validation
        stop_signal = torch.tensor([0], dtype=torch.int, device=device)

        if is_main_process():
            val_metrics = evaluate(
                model.module, val_loader, criterion, physics_extractor, device,
                use_mae, use_gating, use_vdr,
            )
        else:
            val_metrics = {'loss': 0, 'soh_rmse': 0, 'soh_mae': 0, 'soh_r2': 0,
                          'vdr_rmse': 0, 'mae_recon_rmse': 0}

        dist.barrier()
        elapsed = time.time() - t_epoch
        epoch_times.append(elapsed)

        if is_main_process():
            avg_epoch_time = np.mean(epoch_times[-10:])
            eta_seconds = avg_epoch_time * (args.epochs - epoch)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            is_best = val_metrics['loss'] < best_val_loss
            marker = " *BEST*" if is_best else ""
            if val_metrics['soh_rmse'] < best_soh_rmse:
                best_soh_rmse = val_metrics['soh_rmse']

            print(f" {epoch:>3d}/{args.epochs:<3d} | "
                  f"{train_metrics['total']:>10.4f} | "
                  f"{val_metrics['loss']:>10.4f} | "
                  f"{val_metrics['soh_rmse']:>9.4f} | "
                  f"{val_metrics['vdr_rmse']:>9.4f} | "
                  f"{val_metrics['mae_recon_rmse']:>9.4f} | "
                  f"{val_metrics['soh_r2']:>6.3f} | "
                  f"{elapsed:>5.1f}s | "
                  f"{eta_str:>10s}"
                  f"{marker}", flush=True)

            history.append({
                'epoch': epoch, 'train_loss': train_metrics['total'],
                'train_loss_mae': train_metrics['mae'],
                'train_loss_soh': train_metrics['soh'],
                'train_loss_vdr': train_metrics['vdr'],
                'val_loss': val_metrics['loss'],
                'val_soh_rmse': val_metrics['soh_rmse'],
                'val_soh_mae': val_metrics['soh_mae'],
                'val_vdr_rmse': val_metrics['vdr_rmse'],
                'val_mae_recon_rmse': val_metrics['mae_recon_rmse'],
                'val_soh_r2': val_metrics['soh_r2'],
                'lr': current_lr, 'epoch_time': elapsed,
            })

            if is_best:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_soh_rmse': val_metrics['soh_rmse'],
                    'args': vars(args),
                }, os.path.join(args.save_dir, f'{ckpt_name}_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print_rank0("-" * 80)
                    print_rank0(f"  >> Early stopping at epoch {epoch}")
                    stop_signal += 1

        # DDP Sync: Broadcast the stop signal from Rank 0 to all other ranks
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    total_time = time.time() - train_start

    # --- Test (Rank 0 only) ---
    if is_main_process():
        print_rank0(f"\n  Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
        print_rank0("\n[4/4] Final Testing (Pooled + Per-Dataset)...")
        ckpt = torch.load(os.path.join(args.save_dir, f'{ckpt_name}_best.pt'),
                          map_location=device)
        model.module.load_state_dict(ckpt['model_state_dict'])

        # return_per_sample=True: also collect cell_id for per-dataset breakdown
        test_metrics = evaluate(
            model.module, test_loader, criterion, physics_extractor, device,
            use_mae, use_gating, use_vdr, return_per_sample=True,
        )

        print_rank0()
        print_rank0("=" * 70)
        print_rank0("  FINAL TEST RESULTS (DDP, Multi-Task PG-M2TN)")
        print_rank0(f"  Ablation: {args.ablation}")
        print_rank0("=" * 70)

        task_display = [
            ("Task 1: SOH Prediction (Primary)",  "soh"),
            ("Task 2: VDR Prediction (Auxiliary)", "vdr"),
        ]
        for name, prefix in task_display:
            rmse_v = test_metrics[f'{prefix}_rmse']
            mae_v = test_metrics[f'{prefix}_mae']
            mape_v = test_metrics[f'{prefix}_mape']
            mape_f = test_metrics[f'{prefix}_mape_filtered']
            r2_v = test_metrics[f'{prefix}_r2']
            print_rank0(f"  -- {name}")
            print_rank0(f"     RMSE   : {rmse_v:.4f}")
            print_rank0(f"     MAE    : {mae_v:.4f}")
            print_rank0(f"     MAPE   : {mape_v:.2f}%")
            print_rank0(f"     MAPE_f : {mape_f:.2f}%  (SOH>=0.5 only)")
            print_rank0(f"     R2     : {r2_v:.4f}")

        print_rank0("  -- Task 3: MAE Reconstruction")
        print_rank0(f"     RMSE  : {test_metrics['mae_recon_rmse']:.4f}")
        print_rank0(f"     MAE   : {test_metrics['mae_recon_mae']:.4f}")
        print_rank0("  -- Loss Breakdown")
        print_rank0(f"     L_soh : {test_metrics['loss_soh']:.4f}")
        print_rank0(f"     L_vdr : {test_metrics['loss_vdr']:.4f}")
        print_rank0(f"     L_mae : {test_metrics['loss_mae']:.4f}")
        print_rank0(f"     Total : {test_metrics['loss']:.4f}")
        print_rank0("  -- System")
        print_rank0(f"     Params : {count_parameters(model):,}")
        print_rank0(f"     Time   : {str(datetime.timedelta(seconds=int(total_time)))}")
        print_rank0(f"     Done   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_rank0("=" * 70)

        # ── Save raw per-sample data (before pop) ────────────────────
        per_sample_data = test_metrics['_per_sample']
        npz_path = os.path.join(args.save_dir, f'{ckpt_name}_per_sample_test.npz')
        np.savez(npz_path,
                 cell_ids=np.array(per_sample_data['cell_ids'], dtype=str),
                 cycle_idxs=np.array(per_sample_data['cycle_idxs'], dtype=np.int32),
                 soh_pred=np.array(per_sample_data['soh_pred'], dtype=np.float32),
                 soh_true=np.array(per_sample_data['soh_true'], dtype=np.float32),
                 vdr_pred=np.array(per_sample_data['vdr_pred'], dtype=np.float32),
                 vdr_true=np.array(per_sample_data['vdr_true'], dtype=np.float32))
        print_rank0(f"  Per-sample NPZ saved: {npz_path}")

        # ── Per-dataset breakdown (Table S1) ─────────────────────────
        per_ds = compute_per_dataset_metrics(test_metrics.pop('_per_sample'))

        print_rank0()
        print_rank0("  PER-DATASET BREAKDOWN")
        print_rank0(f"  {'Dataset':<12} {'N_samp':>7} {'SOH RMSE':>9} {'SOH MAE':>8} "
                    f"{'MAPE':>7} {'MAPE_f':>7} {'SOH R²':>7} {'VDR RMSE':>9}")
        print_rank0("  " + "-" * 78)
        for ds, m in per_ds.items():
            if ds.startswith('_'):
                continue
            print_rank0(
                f"  {ds:<12} {m['n_samples']:>7d} "
                f"{m['soh']['rmse']:>9.4f} {m['soh']['mae']:>8.4f} "
                f"{m['soh']['mape']:>6.2f}% {m['soh']['mape_filtered']:>6.2f}% "
                f"{m['soh']['r2']:>7.4f} "
                f"{m['vdr']['rmse']:>9.4f}"
            )
        if '_macro_avg' in per_ds:
            ma = per_ds['_macro_avg']
            print_rank0("  " + "-" * 68)
            print_rank0(f"  {'Macro-avg':<12} {'':>7} "
                        f"{ma['soh']['rmse']:>9.4f} {'':>8} {'':>9} {'':>7} "
                        f"{ma['vdr']['rmse']:>9.4f}")
        print_rank0("=" * 70)

        results = {
            'ablation': args.ablation,
            'test_loss': test_metrics['loss'],
            # ── Pooled metrics (Table 1 main) ──
            'test_soh_rmse':  test_metrics['soh_rmse'],
            'test_soh_mae':   test_metrics['soh_mae'],
            'test_soh_mape':  test_metrics['soh_mape'],
            'test_soh_mape_filtered': test_metrics['soh_mape_filtered'],
            'test_soh_r2':    test_metrics['soh_r2'],
            'test_vdr_rmse':  test_metrics['vdr_rmse'],
            'test_vdr_mae':   test_metrics['vdr_mae'],
            'test_vdr_mape':  test_metrics['vdr_mape'],
            'test_vdr_mape_filtered': test_metrics['vdr_mape_filtered'],
            'test_vdr_r2':    test_metrics['vdr_r2'],
            'test_mae_recon_rmse': test_metrics['mae_recon_rmse'],
            'test_mae_recon_mae':  test_metrics['mae_recon_mae'],
            'test_loss_soh':  test_metrics['loss_soh'],
            'test_loss_vdr':  test_metrics['loss_vdr'],
            'test_loss_mae':  test_metrics['loss_mae'],
            # ── Per-dataset breakdown (Table S1) ──
            'per_dataset': per_ds,
            # ── System ──
            'best_val_loss':       best_val_loss,
            'best_soh_rmse':       best_soh_rmse,
            'params':              count_parameters(model),
            'total_train_time_s':  total_time,
            'world_size':          world_size,
            'effective_batch_size':effective_batch,
            'history':             history,
            'args':                vars(args),
        }
        with open(os.path.join(args.save_dir, f'{ckpt_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    cleanup_ddp()


if __name__ == '__main__':
    main()

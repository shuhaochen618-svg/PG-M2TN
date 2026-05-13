"""
Evaluation Metrics
===================
Standardized metric computation for battery prognostics:
  - RMSE  : Root Mean Square Error
  - MAE   : Mean Absolute Error
  - MAPE  : Mean Absolute Percentage Error
  - R²    : Coefficient of Determination
"""

import numpy as np


def compute_task_metrics(errors, trues):
    """
    Compute RMSE, MAE, MAPE, and R² for a single prediction task.

    Args:
        errors : 1-D array of (prediction - ground_truth).
        trues  : 1-D array of ground truth values.

    Returns:
        dict with keys: rmse, mae, mape, mape_filtered, r2.
    """
    errors = np.array(errors, dtype=np.float64)
    trues = np.array(trues, dtype=np.float64)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # Standard MAPE (skip near-zero denominators)
    safe = np.abs(trues) > 1e-6
    mape = float(np.mean(np.abs(errors[safe] / trues[safe]))) * 100 \
        if safe.sum() > 0 else 0.0

    # Filtered MAPE: excludes end-of-life samples (|true| < 0.5)
    # to avoid denominator explosion — standard in battery literature.
    practical = np.abs(trues) >= 0.5
    safe_p = practical & safe
    mape_filtered = float(np.mean(np.abs(errors[safe_p] / trues[safe_p]))) * 100 \
        if safe_p.sum() > 0 else 0.0

    # R² (guard against near-zero variance)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot >= 1e-4 else 0.0

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mape_filtered': mape_filtered,
        'r2': r2,
    }


def get_dataset_prefix(cell_id, known=('CALCE', 'HUST', 'HNEI', 'CALB', 'ISU_ILCC')):
    """
    Infer dataset name from cell_id string.

    Handles naming variants like 'HUST_1-2' → 'HUST',
    'ISU-ILCC_G10C4' → 'ISU_ILCC', 'CALCE_CS2_35' → 'CALCE'.
    """
    uid = str(cell_id).upper().replace('-', '_')
    for ds in known:
        if uid.startswith(ds):
            return ds
    for ds in known:
        if ds in uid:
            return ds
    return 'Unknown'


def compute_per_dataset_metrics(per_sample):
    """
    Group per-sample predictions by dataset prefix and compute metrics.

    Args:
        per_sample : dict with keys 'cell_ids', 'soh_pred', 'soh_true',
                     'vdr_pred', 'vdr_true' (each a list).

    Returns:
        dict mapping dataset names to their metrics, plus '_macro_avg'.
    """
    from collections import defaultdict

    groups = defaultdict(lambda: {
        'soh_p': [], 'soh_t': [], 'vdr_p': [], 'vdr_t': []})

    for cid, sp, st, vp, vt in zip(
            per_sample['cell_ids'],
            per_sample['soh_pred'], per_sample['soh_true'],
            per_sample['vdr_pred'], per_sample['vdr_true']):
        ds = get_dataset_prefix(cid)
        groups[ds]['soh_p'].append(sp)
        groups[ds]['soh_t'].append(st)
        groups[ds]['vdr_p'].append(vp)
        groups[ds]['vdr_t'].append(vt)

    out = {}
    soh_rmses, vdr_rmses = [], []
    for ds, g in sorted(groups.items()):
        soh_p = np.array(g['soh_p'])
        soh_t = np.array(g['soh_t'])
        vdr_p = np.array(g['vdr_p'])
        vdr_t = np.array(g['vdr_t'])
        soh_m = compute_task_metrics(soh_p - soh_t, soh_t)
        vdr_m = compute_task_metrics(vdr_p - vdr_t, vdr_t)
        out[ds] = {'n_samples': len(soh_p), 'soh': soh_m, 'vdr': vdr_m}
        soh_rmses.append(soh_m['rmse'])
        vdr_rmses.append(vdr_m['rmse'])

    if soh_rmses:
        out['_macro_avg'] = {
            'soh': {'rmse': float(np.mean(soh_rmses))},
            'vdr': {'rmse': float(np.mean(vdr_rmses))},
            'note': 'Unweighted mean across datasets',
        }
    return out

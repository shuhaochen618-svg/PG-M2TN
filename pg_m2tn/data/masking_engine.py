"""
PG-M2TN Data Engine — Masking Engine
======================================
Simulates real-world IoT data fragmentation by applying continuous-block
masking to battery cycle sequences.

Supports:
  - Random continuous masking (block mask)
  - Configurable fixed or random masking ratios [0.1, 0.9]
  - Deterministic masking for reproducible evaluation (via seed)
  - Returns both masked input and original for MAE reconstruction loss
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MaskedBatteryDataset(Dataset):
    """
    Wraps a BatteryCycleDataset and applies continuous-block masking
    to simulate real-world fragmented charging data.

    Args:
        base_dataset     : Instance of BatteryCycleDataset (or Subset).
        fixed_mask_ratio : If set, uses this ratio for all samples (evaluation).
                           If None, samples uniformly from [min_ratio, max_ratio].
        mask_value       : Value to fill masked positions (default: 0.0).
        min_ratio        : Minimum masking ratio when sampling randomly.
        max_ratio        : Maximum masking ratio when sampling randomly.
        seed             : Optional seed for reproducible mask positions.
                           Use the same seed across models for fair comparison.
    """

    def __init__(self, base_dataset, fixed_mask_ratio=None, mask_value=0.0,
                 min_ratio=0.1, max_ratio=0.9, seed=None):
        self.base_dataset = base_dataset
        self.fixed_mask_ratio = fixed_mask_ratio
        self.mask_value = mask_value
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        x_full = data['features'].clone()  # [SeqLen, 2]
        seq_len = x_full.shape[0]

        # --- Determine masking ratio ---
        if self.fixed_mask_ratio is not None:
            p = self.fixed_mask_ratio
        else:
            p = np.random.uniform(self.min_ratio, self.max_ratio)

        # --- Create continuous block mask ---
        mask_len = int(seq_len * p)
        x_masked = x_full.clone()
        binary_mask = torch.zeros(seq_len, dtype=torch.bool)

        if mask_len > 0 and mask_len < seq_len:
            if self.seed is not None:
                rng_state = np.random.RandomState(self.seed + idx)
                start_idx = rng_state.randint(0, seq_len - mask_len)
            else:
                start_idx = np.random.randint(0, seq_len - mask_len)
            x_masked[start_idx:start_idx + mask_len, :] = self.mask_value
            binary_mask[start_idx:start_idx + mask_len] = True

        return {
            'x_full': x_full,           # [SeqLen, 2] Original complete sequence
            'x_masked': x_masked,       # [SeqLen, 2] Masked (fragmented) sequence
            'binary_mask': binary_mask,  # [SeqLen]    True = masked position
            'mask_ratio': float(p),
            'soh': data['soh'],
            'vdr': data['vdr'],
            'V_raw': data['V_raw'],
            'Q_raw': data['Q_raw'],
            'cycle_idx': data['cycle_idx'],
            'cell_id': data['cell_id'],
        }

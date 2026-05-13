"""
PG-M2TN Data Engine — Battery Cycle Dataset Loader
====================================================
Loads battery cycling data from unified .pkl format
(CALCE, HUST, HNEI, CALB, ISU_ILCC).

Extracts [V, I] time-series sequences and cycle-level health labels (SOH, VDR).
"""

import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helper: Robust Sequence Resampling
# ---------------------------------------------------------------------------
def resample_sequence(arr, target_len):
    """Resample a 1-D array to a fixed length using linear interpolation."""
    if arr is None or len(arr) == 0:
        return np.zeros(target_len)
    arr = np.array(arr, dtype=np.float64)
    # Replace NaN with forward/backward fill, then zero
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros(target_len)
    if not np.all(mask):
        arr = np.interp(np.arange(len(arr)), np.where(mask)[0], arr[mask])
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, arr, kind='linear', fill_value='extrapolate')
    return f(x_new).astype(np.float32)


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------
class BatteryCycleDataset(Dataset):
    """
    Builds cycle-level samples from raw .pkl battery data.

    Each sample is a dictionary:
        features  : [seq_len, 2]  (V, I) resampled and normalized time-series
        V_raw     : [seq_len]     raw voltage for IC-curve extraction
        Q_raw     : [seq_len]     raw capacity for IC-curve extraction
        soh       : float         SOH = Q_discharge_max / Q_reference
        vdr       : float         Voltage Distortion Ratio (normalized CV)
        cycle_idx : int           cycle index within the cell
        cell_id   : str           cell identifier

    Expected .pkl format:
        {
            'cell_id': str,
            'nominal_capacity_in_Ah': float,
            'cathode_material': str,  # optional
            'cycle_data': [
                {
                    'voltage_in_V': array,
                    'current_in_A': array,
                    'charge_capacity_in_Ah': array,
                    'discharge_capacity_in_Ah': array,
                },
                ...
            ]
        }

    Args:
        data_root  : Path to the dataset directory containing sub-folders.
        datasets   : List of dataset names to load (e.g., ['CALCE', 'HUST']).
        seq_len    : Fixed sequence length after resampling (default: 512).
        min_cycles : Skip cells with fewer cycles than this (default: 50).
    """

    def __init__(self, data_root, datasets=None, seq_len=512, min_cycles=50):
        super().__init__()
        self.seq_len = seq_len
        self.samples = []
        self.cell_meta = {}  # cell_id -> {nominal_cap, cathode, ...}

        if datasets is None:
            datasets = ['CALCE', 'CALB', 'HNEI', 'HUST', 'ISU_ILCC']

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731

        for ds_name in datasets:
            ds_path = os.path.join(data_root, ds_name)
            if not os.path.isdir(ds_path):
                print(f"[WARN] Dataset folder not found: {ds_path}")
                continue
            pkl_files = sorted(glob.glob(os.path.join(ds_path, '*.pkl')))
            for pkl_path in tqdm(pkl_files, desc=f"Loading {ds_name}", leave=False):
                self._load_cell(pkl_path, min_cycles, ds_name=ds_name)

        print(f"[BatteryCycleDataset] Loaded {len(self.samples)} samples "
              f"from {len(self.cell_meta)} cells.")

    def _load_cell(self, pkl_path, min_cycles, ds_name=None):
        """Load a single cell .pkl and build per-cycle samples."""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        cell_id = data['cell_id']
        # Ensure cell_id is prefixed with dataset name
        if ds_name and not cell_id.upper().replace('-', '_').startswith(ds_name.upper()):
            cell_id = f"{ds_name}_{cell_id}"
        nom_cap = data.get('nominal_capacity_in_Ah', 1.0)
        cycles = data['cycle_data']

        if len(cycles) < min_cycles:
            return

        self.cell_meta[cell_id] = {
            'nominal_capacity': nom_cap,
            'cathode': data.get('cathode_material', 'Unknown'),
            'num_cycles': len(cycles),
        }

        # ------ Pre-compute cycle-level statistics ------
        discharge_caps = []
        for cyc in cycles:
            dc = np.array(cyc['discharge_capacity_in_Ah'], dtype=np.float64)
            discharge_caps.append(np.nanmax(dc) if len(dc) > 0 else 0.0)
        discharge_caps = np.array(discharge_caps)

        # Reference capacity = max of first 5 cycles (robust to outliers)
        ref_cap = np.nanmax(discharge_caps[:5]) if len(discharge_caps) >= 5 \
            else np.nanmax(discharge_caps)
        if ref_cap <= 0 or not np.isfinite(ref_cap):
            ref_cap = nom_cap

        # Pre-compute per-cycle voltage dispersion
        cycle_vdr_list = []
        for cyc in cycles:
            v_cyc = np.array(cyc['voltage_in_V'], dtype=np.float64)
            v_valid = v_cyc[np.isfinite(v_cyc)]
            if len(v_valid) > 10:
                cycle_vdr_list.append(
                    float(np.std(v_valid) / (np.mean(np.abs(v_valid)) + 1e-6)))
            else:
                cycle_vdr_list.append(0.0)

        ref_vdr = np.mean(cycle_vdr_list[:5]) if len(cycle_vdr_list) >= 5 \
            else (cycle_vdr_list[0] if cycle_vdr_list else 0.1)
        if ref_vdr < 0.01:
            ref_vdr = 0.1

        # ------ Build per-cycle samples ------
        for i, cyc in enumerate(cycles):
            v = np.array(cyc['voltage_in_V'], dtype=np.float64)
            c = np.array(cyc['current_in_A'], dtype=np.float64)
            q_charge = np.array(cyc['charge_capacity_in_Ah'], dtype=np.float64)
            q_discharge = np.array(cyc['discharge_capacity_in_Ah'], dtype=np.float64)

            if len(v) < 10:
                continue

            # --- Features: [V, I] resampled and normalized (2 channels) ---
            v_rs = resample_sequence(v, self.seq_len)
            c_rs = resample_sequence(c, self.seq_len)

            # Normalization:
            #   V → [-1, 1] using physics-based bounds [2.5, 4.4] V
            #   I → [-1, 1] using per-sequence max|I|
            V_LOW, V_HIGH = 2.5, 4.4
            v_norm = np.clip((v_rs - V_LOW) / (V_HIGH - V_LOW), 0.0, 1.0) * 2.0 - 1.0
            max_abs_c = np.max(np.abs(c_rs)) + 1e-6
            c_norm = c_rs / max_abs_c

            features = np.stack([v_norm, c_norm], axis=-1)  # [seq_len, 2]

            # --- Raw V, Q for IC curve ---
            q_for_ic = q_charge if np.nanmax(q_charge) > 0 else q_discharge
            v_raw = resample_sequence(v, self.seq_len)
            q_raw = resample_sequence(q_for_ic, self.seq_len)

            # --- Labels ---
            soh = float(np.clip(discharge_caps[i] / ref_cap, 0.0, 1.0))
            vdr = float(np.clip(cycle_vdr_list[i] / ref_vdr, 0.0, 3.0))

            if not np.isfinite(soh):
                soh = 1.0
            if not np.isfinite(vdr):
                vdr = 0.0

            self.samples.append({
                'features': torch.from_numpy(features).float(),
                'V_raw': torch.from_numpy(v_raw).float(),
                'Q_raw': torch.from_numpy(q_raw).float(),
                'soh': torch.tensor(soh, dtype=torch.float32),
                'vdr': torch.tensor(vdr, dtype=torch.float32),
                'cycle_idx': i,
                'cell_id': cell_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Train / Val / Test Split Utility
# ---------------------------------------------------------------------------
def split_by_cell(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset by CELL (not by cycle) to prevent data leakage.

    Uses stratified sampling: cells are grouped by dataset prefix and split
    proportionally within each group to guarantee every dataset has ≥1 cell
    in the test set.

    Args:
        dataset     : BatteryCycleDataset instance.
        train_ratio : Fraction of cells for training (default: 0.7).
        val_ratio   : Fraction of cells for validation (default: 0.15).
        seed        : Random seed for reproducibility.

    Returns:
        train_idx, val_idx, test_idx : Lists of sample indices.
    """
    rng = np.random.RandomState(seed)
    cell_ids = list(dataset.cell_meta.keys())

    known_ds = ('CALCE', 'HUST', 'HNEI', 'CALB', 'ISU_ILCC')

    def _prefix(cid):
        uid = cid.upper().replace('-', '_')
        for ds in known_ds:
            if uid.startswith(ds):
                return ds
        return 'OTHER'

    groups = defaultdict(list)
    for cid in cell_ids:
        groups[_prefix(cid)].append(cid)

    train_cells, val_cells, test_cells = set(), set(), set()
    for ds, cells in groups.items():
        rng.shuffle(cells)
        n = len(cells)
        if n >= 3:
            n_tr = max(1, int(n * train_ratio))
            n_va = max(1, int(n * val_ratio))
            if n_tr + n_va >= n:
                n_tr = max(1, n - 2)
                n_va = 1
        elif n == 2:
            n_tr, n_va = 1, 0
        else:
            n_tr, n_va = 1, 0
        train_cells.update(cells[:n_tr])
        val_cells.update(cells[n_tr:n_tr + n_va])
        test_cells.update(cells[n_tr + n_va:])

    train_idx, val_idx, test_idx = [], [], []
    for i, sample in enumerate(dataset.samples):
        cid = sample['cell_id']
        if cid in train_cells:
            train_idx.append(i)
        elif cid in val_cells:
            val_idx.append(i)
        else:
            test_idx.append(i)

    for ds in sorted(groups.keys()):
        n_test_ds = sum(1 for c in groups[ds] if c in test_cells)
        print(f"  [Split] {ds}: {len(groups[ds])} cells → test={n_test_ds}")

    print(f"[Split] Train: {len(train_idx)} samples ({len(train_cells)} cells) | "
          f"Val: {len(val_idx)} samples ({len(val_cells)} cells) | "
          f"Test: {len(test_idx)} samples ({len(test_cells)} cells)")
    return train_idx, val_idx, test_idx

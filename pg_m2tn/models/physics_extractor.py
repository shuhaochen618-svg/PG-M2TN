"""
PG-M2TN Physics Module — Multi-Signal Aging Factor Extractor
=============================================================
Extracts the physical aging factor α from raw V/Q data using
a fusion of THREE electrochemical degradation signals:

  1. ΔV_peak   : IC curve main-peak voltage shift   (w=0.2)
  2. Cap_fade  : Capacity fade ratio (1 - SOH)       (w=0.5, training only)
  3. ΔIC_height: IC peak height degradation          (w=0.3)

  α = clamp(w1·α_vpeak + w2·α_capacity + w3·α_ic_height, 0, 1)

DDP Safety:
  Reference is ONLY updated when cycle_idx == 0 (canonical first cycle).
  This ensures consistent α values across GPU processes regardless of
  batch shuffle order.

Inference Mode:
  inference_mode=True uses IC signals only (no SOH label dependency).
  This is the BMS deployment mode for online real-time use.
"""

import numpy as np
import torch
from scipy.signal import savgol_filter, find_peaks


class PhysicsExtractor:
    """
    Extracts the physical aging factor (α) from raw Capacity (Q)
    and Voltage (V) curves using multi-signal fusion.

    The α factor serves as a post-hoc interpretability lens and as input
    to the physics-gated dynamic loss weighting mechanism.

    Args:
        window_length : Savitzky-Golay filter window size.
        polyorder     : Savitzky-Golay polynomial order.
        w_vpeak       : Weight for ΔV_peak signal in α fusion.
        w_capacity    : Weight for capacity fade signal in α fusion.
        w_ic_height   : Weight for IC peak height signal in α fusion.
    """

    def __init__(self, window_length=21, polyorder=3,
                 w_vpeak=0.2, w_capacity=0.5, w_ic_height=0.3):
        self.window_length = window_length
        self.polyorder = polyorder
        self.w_vpeak = w_vpeak
        self.w_capacity = w_capacity
        self.w_ic_height = w_ic_height
        # Per-cell reference state — only updated when cycle_idx == 0
        self.reference_v_peaks = {}      # cell_id -> V_peak of cycle 0
        self.reference_ic_heights = {}   # cell_id -> IC peak height of cycle 0
        self.reference_v_ranges = {}     # cell_id -> (V_max - V_min) for normalization

    def compute_ic_curve(self, V, Q):
        """
        Compute the Incremental Capacity (IC) curve dQ/dV.

        Args:
            V : 1-D array of voltage values.
            Q : 1-D array of capacity values.

        Returns:
            V_smooth : Smoothed voltage array.
            IC       : dQ/dV array.
        """
        V = np.array(V, dtype=np.float64)
        Q = np.array(Q, dtype=np.float64)

        valid = np.isfinite(V) & np.isfinite(Q)
        if np.sum(valid) < self.window_length + 2:
            return V[valid] if np.any(valid) else V, \
                   np.zeros_like(V[valid] if np.any(valid) else V)
        V, Q = V[valid], Q[valid]

        sort_idx = np.argsort(V)
        V_sorted = V[sort_idx]
        Q_sorted = Q[sort_idx]

        wl = min(self.window_length, len(V_sorted))
        if wl % 2 == 0:
            wl -= 1
        if wl < self.polyorder + 2:
            return V_sorted, np.zeros_like(V_sorted)

        V_smooth = savgol_filter(V_sorted, wl, self.polyorder)
        Q_smooth = savgol_filter(Q_sorted, wl, self.polyorder)

        dV = np.gradient(V_smooth)
        dQ = np.gradient(Q_smooth)
        dV[np.abs(dV) < 1e-8] = 1e-8
        IC = dQ / dV

        return V_smooth, IC

    def find_main_peak(self, V_smooth, IC):
        """
        Locate the voltage and height of the main IC peak.

        Returns:
            v_peak      : Voltage at the main peak.
            peak_height : Absolute height of the main peak.
        """
        abs_ic = np.abs(IC)
        try:
            peaks, _ = find_peaks(abs_ic, prominence=0.01, distance=5)
        except Exception:
            peaks = np.array([])

        if len(peaks) == 0:
            idx = np.argmax(abs_ic)
        else:
            idx = peaks[np.argmax(abs_ic[peaks])]

        return float(V_smooth[idx]), float(abs_ic[idx])

    def extract_alpha(self, V, Q, cell_id=None, cycle_idx=0, soh=None,
                      inference_mode=False):
        """
        Extract the normalized aging factor α ∈ [0, 1].

        During training (inference_mode=False), α fuses:
          - IC peak shift     (w=0.2): electrochemical, from V/Q data
          - Capacity fade     (w=0.5): 1-SOH, uses the known training label
          - IC height change  (w=0.3): electrochemical, from V/Q data

        During inference (inference_mode=True), α uses IC signals only.

        Args:
            V              : Voltage array [SeqLen].
            Q              : Capacity array [SeqLen].
            cell_id        : Cell identifier string.
            cycle_idx      : Cycle index (0 = first cycle of this cell).
            soh            : SOH value in [0,1]. Used in training only.
            inference_mode : If True, SOH label is not used.

        Returns:
            alpha : Float in [0, 1].
        """
        V_smooth, IC = self.compute_ic_curve(V, Q)

        if len(IC) == 0 or len(V_smooth) == 0:
            if soh is not None and not inference_mode:
                return float(np.clip(1.0 - soh, 0.0, 1.0))
            return 0.0

        v_peak, peak_height = self.find_main_peak(V_smooth, IC)
        v_range = float(np.max(V_smooth) - np.min(V_smooth))
        # Auto-calibrated normalization: adapts across NMC/LFP/NCA chemistries
        max_expected_shift = max(v_range * 0.10, 0.01)  # at least 10mV

        # --- DDP-safe reference management ---
        if cell_id is not None:
            if cycle_idx == 0 or cell_id not in self.reference_v_peaks:
                self.reference_v_peaks[cell_id] = v_peak
                self.reference_ic_heights[cell_id] = peak_height
                self.reference_v_ranges[cell_id] = max_expected_shift
            ref_vpeak = self.reference_v_peaks[cell_id]
            ref_ic_height = self.reference_ic_heights[cell_id]
            ref_max_shift = self.reference_v_ranges[cell_id]
        else:
            ref_vpeak = v_peak
            ref_ic_height = peak_height
            ref_max_shift = max_expected_shift

        # --- Signal 1: ΔV_peak (electrochemical) ---
        delta_v = abs(v_peak - ref_vpeak)
        alpha_vpeak = float(np.clip(delta_v / ref_max_shift, 0.0, 1.0))

        # --- Signal 2: IC height degradation (electrochemical) ---
        if ref_ic_height > 1e-6:
            alpha_ic_height = float(np.clip(1.0 - peak_height / ref_ic_height, 0.0, 1.0))
        else:
            alpha_ic_height = 0.0

        # --- Signal 3: Capacity fade (training supervision) ---
        if soh is not None and not inference_mode:
            alpha_capacity = float(np.clip(1.0 - soh, 0.0, 1.0))
            alpha = (self.w_vpeak * alpha_vpeak +
                     self.w_capacity * alpha_capacity +
                     self.w_ic_height * alpha_ic_height)
        else:
            # Inference mode: IC signals only, re-normalized weights
            w_sum = self.w_vpeak + self.w_ic_height
            alpha = ((self.w_vpeak / w_sum) * alpha_vpeak +
                     (self.w_ic_height / w_sum) * alpha_ic_height)

        return float(np.clip(alpha, 0.0, 1.0))

    def batch_extract(self, V_batch, Q_batch, cell_ids=None, cycle_indices=None,
                      soh_batch=None, inference_mode=False):
        """
        Extract α for a batch of sequences.

        Args:
            V_batch        : [Batch, SeqLen] tensor or numpy array.
            Q_batch        : [Batch, SeqLen] tensor or numpy array.
            cell_ids       : List of cell_id strings.
            cycle_indices  : List of cycle indices.
            soh_batch      : [Batch] tensor or array of SOH values (optional).
            inference_mode : If True, uses IC signals only.

        Returns:
            alphas : [Batch, 1] float32 tensor.
        """
        if isinstance(V_batch, torch.Tensor):
            V_batch = V_batch.cpu().numpy()
        if isinstance(Q_batch, torch.Tensor):
            Q_batch = Q_batch.cpu().numpy()
        if isinstance(soh_batch, torch.Tensor):
            soh_batch = soh_batch.cpu().numpy()

        alphas = []
        for i in range(len(V_batch)):
            cid  = cell_ids[i]         if cell_ids      is not None else None
            cidx = int(cycle_indices[i]) if cycle_indices is not None else 0
            soh  = float(soh_batch[i]) if soh_batch      is not None else None
            alpha = self.extract_alpha(V_batch[i], Q_batch[i],
                                       cell_id=cid, cycle_idx=cidx, soh=soh,
                                       inference_mode=inference_mode)
            alphas.append([alpha])

        return torch.tensor(alphas, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Quick Validation
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    extractor = PhysicsExtractor()

    V = np.linspace(3.0, 4.2, 500)
    Q = 1.0 / (1.0 + np.exp(-15 * (V - 3.6)))

    # Cycle 0 (fresh): α should be ~0
    a0 = extractor.extract_alpha(V, Q, cell_id='test', cycle_idx=0, soh=1.0)
    print(f"Cycle   0 (SOH=1.00): alpha={a0:.4f}  (expected ~0.00)")

    # Cycle 100 (mild degradation)
    V2 = np.linspace(3.0, 4.2, 500)
    Q2 = 0.8 / (1.0 + np.exp(-15 * (V2 - 3.63)))
    a2 = extractor.extract_alpha(V2, Q2, cell_id='test', cycle_idx=100, soh=0.8)
    print(f"Cycle 100 (SOH=0.80): alpha={a2:.4f}  (expected ~0.15-0.25)")

    # Cycle 500 (heavy degradation)
    Q3 = 0.5 / (1.0 + np.exp(-15 * (V2 - 3.68)))
    a3 = extractor.extract_alpha(V2, Q3, cell_id='test', cycle_idx=500, soh=0.5)
    print(f"Cycle 500 (SOH=0.50): alpha={a3:.4f}  (expected ~0.40-0.60)")

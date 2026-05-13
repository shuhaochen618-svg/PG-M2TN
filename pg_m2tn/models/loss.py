"""
PG-M2TN Loss Module — Physics-Gated Dynamic Loss Function
==========================================================
The joint multi-task optimization objective for PG-M2TN:
  - Self-supervised MAE reconstruction loss (L_mae)
  - Physics-gated dynamic task weighting for SOH and VDR
  - Supports ablation: fixed uniform weights when gating is disabled

Weight Formulas (Lower-Bounded Batch-Level Gating):
  alpha_mean = mean(alpha)                         # batch-level aggregation
  w_SOH = 0.50 + 0.20 * (1.0 - alpha_mean)        # ∈ [0.50, 0.70]
  w_VDR = 0.50 + 0.20 * alpha_mean                 # ∈ [0.50, 0.70]

Physical Curriculum:
  - Early-life batches (alpha ≈ 0): w_SOH=0.70, w_VDR=0.50
    → Focus on SOH trajectory learning, maintain baseline VDR tracking.
  - Late-life batches (alpha ≈ 1): w_SOH=0.50, w_VDR=0.70
    → Focus on VDR physical signals, maintain baseline SOH tracking.
  - Both tasks are ALWAYS guaranteed ≥ 0.50 weight (no starvation).

Ablation (No-Gating) uses FIXED 0.50/0.50 weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGatedLoss(nn.Module):
    """
    Physics-Gated Dynamic Loss for PG-M2TN.

    Combines three loss components:
      1. L_mae : MSE reconstruction loss (self-supervised regularizer)
      2. L_soh : Weighted MSE for SOH prediction (primary task)
      3. L_vdr : Weighted MSE for VDR prediction (auxiliary task)

    The weighting between SOH and VDR adapts dynamically based on the
    physical aging factor (alpha), implementing a curriculum that shifts
    focus from SOH in early life to VDR in late life.

    Args:
        lambda_mae     : Coefficient for the MAE reconstruction loss.
        scale_vdr      : Scaling factor to balance VDR vs SOH loss magnitude.
        min_soh_weight : Reserved for API compatibility.
        min_vdr_weight : Reserved for API compatibility.
    """

    def __init__(self, lambda_mae: float = 1.0, scale_vdr: float = 1.0,
                 min_soh_weight: float = 0.50, min_vdr_weight: float = 0.10):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.scale_vdr = scale_vdr
        self.min_soh_weight = min_soh_weight
        self.min_vdr_weight = min_vdr_weight

    @staticmethod
    def compute_dynamic_weights(alpha, min_soh_weight=0.50, min_vdr_weight=0.10):
        """
        Compute physics-gated task weights from aging factor.

        Uses batch-level aggregation with lower-bounded gating to prevent
        task starvation. Both tasks are guaranteed ≥ 0.50 weight.

        Args:
            alpha          : [Batch, 1] tensor, values in [0, 1].
            min_soh_weight : Reserved for API compatibility.
            min_vdr_weight : Reserved for API compatibility.

        Returns:
            w_soh, w_vdr : each [Batch, 1], uniform across the batch.
        """
        alpha_mean = alpha.mean()  # scalar

        w_soh_val = 0.50 + 0.20 * (1.0 - alpha_mean)
        w_vdr_val = 0.50 + 0.20 * alpha_mean

        w_soh = torch.full_like(alpha, w_soh_val.item())
        w_vdr = torch.full_like(alpha, w_vdr_val.item())
        return w_soh, w_vdr

    def forward(
        self,
        x_full, x_recon,
        soh_true, soh_pred,
        vdr_true, vdr_pred,
        alpha,
        use_dynamic_gating: bool = True,
        use_mae: bool = True,
        use_vdr: bool = True,
    ):
        """
        Compute the total joint loss.

        Args:
            x_full    : [B, L, C] Original unmasked sequence.
            x_recon   : [B, L, C] Reconstructed sequence (None if MAE disabled).
            soh_true  : [B] or [B,1] Ground truth SOH.
            soh_pred  : [B, 1] Predicted SOH.
            vdr_true  : [B] or [B,1] Ground truth VDR.
            vdr_pred  : [B, 1] Predicted VDR.
            alpha     : [B, 1] Physical aging factor.
            use_dynamic_gating : If False, uses fixed balanced weights (ablation).
            use_mae            : If False, skips MAE loss (ablation).
            use_vdr            : If False, skips VDR loss (ablation).

        Returns:
            total_loss : Scalar tensor.
            loss_dict  : Dict of individual loss components (detached floats).
        """
        # --- 1. MAE Reconstruction Loss ---
        if use_mae and x_recon is not None:
            l_mae = F.mse_loss(x_recon, x_full)
        else:
            l_mae = torch.tensor(0.0, device=soh_pred.device)

        # --- 2. Task Losses (per-sample for weighted reduction) ---
        if soh_true.dim() == 1:
            soh_true = soh_true.unsqueeze(1)
        if vdr_true.dim() == 1:
            vdr_true = vdr_true.unsqueeze(1)

        l_soh = F.mse_loss(soh_pred, soh_true, reduction='none')  # [B, 1]
        l_vdr = F.mse_loss(vdr_pred, vdr_true, reduction='none') * self.scale_vdr

        # --- 3. Dynamic Weight Assignment ---
        if use_dynamic_gating:
            w_soh, w_vdr = self.compute_dynamic_weights(
                alpha, self.min_soh_weight, self.min_vdr_weight)
        else:
            # Ablation (No-Gating): fixed 0.50/0.50 weights
            ones = torch.ones_like(alpha)
            w_soh = ones * 0.50
            w_vdr = ones * 0.50

        weighted_l_soh = (w_soh * l_soh).mean()
        weighted_l_vdr = (w_vdr * l_vdr).mean()

        # --- 4. Total Joint Loss ---
        if use_vdr:
            total_loss = self.lambda_mae * l_mae + weighted_l_soh + weighted_l_vdr
        else:
            total_loss = self.lambda_mae * l_mae + weighted_l_soh

        loss_dict = {
            'total': total_loss.item(),
            'mae': l_mae.item(),
            'soh': weighted_l_soh.item(),
            'vdr': weighted_l_vdr.item(),
            'w_soh_mean': w_soh.mean().item(),
            'w_vdr_mean': w_vdr.mean().item(),
        }

        return total_loss, loss_dict

"""
PG-M2TN Core Network — Physics-Guided Masked Multi-Task Network
================================================================
Architecture:
  1. Shared Encoder  : BiLSTM backbone (lightweight, edge-friendly)
  2. MAE Decoder     : MLP-based sequence reconstruction head
  3. Multi-Task Heads: 2 parallel MLP heads for SOH, VDR

Design choices:
  - BiLSTM chosen over Transformer for lower FLOPs → edge deployment
  - Two-layer MLP attention pooling for physically meaningful time-step weighting
  - Supports ablation flags to disable MAE or individual heads

Reference:
  "Bridging Microscopic Polarization and Macroscopic Degradation:
   A Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics"
"""

import torch
import torch.nn as nn


class PGM2TN(nn.Module):
    """
    Physics-Guided Masked Multi-Task Network (PG-M2TN).

    A compact, edge-deployable architecture that concurrently predicts:
      - SOH (State of Health): macroscopic capacity degradation
      - VDR (Voltage Distortion Ratio): microscopic polarization indicator

    The optional MAE decoder reconstructs masked input sequences as a
    self-supervised regularizer that prevents latent-space collapse.

    Args:
        input_dim  : Number of input channels (V, I) = 2.
        hidden_dim : BiLSTM hidden size per direction (default: 128).
        num_layers : Number of stacked LSTM layers (default: 2).
        dropout    : Dropout rate (default: 0.2).
        enable_mae : If False, disables the MAE decoder (for ablation studies).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        enable_mae: bool = True,
    ):
        super().__init__()
        self.enable_mae = enable_mae
        self.hidden_dim = hidden_dim
        enc_out_dim = hidden_dim * 2  # bidirectional

        # ============================================================
        # 1. Shared Encoder: Bidirectional LSTM
        # ============================================================
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(enc_out_dim)

        # ============================================================
        # 2. MAE Decoder: Reconstruct full [V, I] sequence
        # ============================================================
        if enable_mae:
            self.mae_decoder = nn.Sequential(
                nn.Linear(enc_out_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
            )

        # ============================================================
        # 3. Multi-Task Prediction Heads
        # ============================================================
        # Two-layer MLP attention pooling: more expressive than a single
        # linear layer, better at identifying physically meaningful time
        # steps (e.g., voltage plateau). Tanh activation keeps weights
        # in (-1,1) before softmax normalization.
        self.attn_pool = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim // 2),
            nn.Tanh(),
            nn.Linear(enc_out_dim // 2, 1),
        )

        self.soh_head = nn.Sequential(
            nn.Linear(enc_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.vdr_head = nn.Sequential(
            nn.Linear(enc_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _attention_pool(self, enc_out):
        """Attention-weighted pooling over time steps."""
        # enc_out: [B, L, D]
        attn_weights = torch.softmax(self.attn_pool(enc_out), dim=1)  # [B, L, 1]
        pooled = (enc_out * attn_weights).sum(dim=1)  # [B, D]
        return pooled

    def forward(self, x_masked):
        """
        Forward pass.

        Args:
            x_masked: [Batch, SeqLen, InputDim] Masked input sequence.

        Returns:
            x_recon  : [Batch, SeqLen, InputDim] Reconstructed sequence (or None).
            soh_pred : [Batch, 1] SOH prediction.
            vdr_pred : [Batch, 1] VDR prediction.
        """
        # --- Encoder ---
        enc_out, _ = self.encoder(x_masked)  # [B, L, hidden*2]
        enc_out = self.layer_norm(enc_out)

        # --- MAE Decoder ---
        if self.enable_mae:
            x_recon = self.mae_decoder(enc_out)  # [B, L, input_dim]
        else:
            x_recon = None

        # --- Attention Pooling ---
        global_state = self._attention_pool(enc_out)  # [B, hidden*2]

        # --- Task Heads ---
        soh_pred = self.soh_head(global_state)  # [B, 1]
        vdr_pred = self.vdr_head(global_state)  # [B, 1]

        return x_recon, soh_pred, vdr_pred

    def forward_with_interpretability(self, x_masked):
        """
        Forward pass that also returns interpretability artifacts.

        Returns a dict containing:
          - x_recon      : [B, L, InputDim] Reconstructed sequence
          - soh_pred     : [B, 1]           SOH prediction
          - vdr_pred     : [B, 1]           VDR prediction
          - attn_weights : [B, L, 1]        Attention pooling weights
          - enc_hidden   : [B, L, D]        Encoder hidden states (for t-SNE)
          - global_state : [B, D]           Pooled representation (for latent viz)
        """
        enc_out, _ = self.encoder(x_masked)
        enc_out = self.layer_norm(enc_out)

        if self.enable_mae:
            x_recon = self.mae_decoder(enc_out)
        else:
            x_recon = None

        attn_weights = torch.softmax(self.attn_pool(enc_out), dim=1)  # [B, L, 1]
        global_state = (enc_out * attn_weights).sum(dim=1)

        soh_pred = self.soh_head(global_state)
        vdr_pred = self.vdr_head(global_state)

        return {
            'x_recon': x_recon,
            'soh_pred': soh_pred,
            'vdr_pred': vdr_pred,
            'attn_weights': attn_weights.detach(),   # [B, L, 1]
            'enc_hidden': enc_out.detach(),           # [B, L, D]
            'global_state': global_state.detach(),    # [B, D]
        }


# ---------------------------------------------------------------------------
# Model Summary Utility
# ---------------------------------------------------------------------------
def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = PGM2TN(input_dim=2, hidden_dim=128, num_layers=2)
    x = torch.randn(4, 512, 2)
    x_recon, soh, vdr = model(x)

    print(f"Input:      {x.shape}")
    print(f"Recon:      {x_recon.shape}")
    print(f"SOH pred:   {soh.shape}")
    print(f"VDR pred:   {vdr.shape}")
    print(f"Parameters: {count_parameters(model):,}")

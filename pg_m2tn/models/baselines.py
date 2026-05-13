"""
Baseline Models for Comparison
================================
Implements the baseline architectures evaluated against PG-M2TN:
  - CNN1D         : 1D Convolutional baseline
  - StandardGRU   : GRU-based sequence model
  - StandardLSTM  : LSTM-based sequence model
  - VanillaTransformer : Standard Transformer encoder
  - HardSharingMTL     : Hard-sharing multi-task learning baseline

All baselines predict SOH only (single-task) for fair comparison.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D CNN baseline for fast local feature extraction."""

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, L, C] -> conv1d expects [B, C, L]
        x = x.transpose(1, 2)
        feat = self.conv(x).squeeze(-1)
        return self.head(feat)


class StandardGRU(nn.Module):
    """Simple GRU baseline (lighter than LSTM)."""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn[-1])


class StandardLSTM(nn.Module):
    """Simple LSTM baseline for SOH prediction."""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1])


class VanillaTransformer(nn.Module):
    """Standard Transformer Encoder baseline."""

    def __init__(self, input_dim=2, d_model=128, n_heads=4, num_layers=4,
                 max_seq_len=1024):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B, L, _ = x.shape
        x_emb = self.embedding(x) + self.pos_emb[:, :L, :]
        out = self.transformer(x_emb)
        pooled = out.mean(dim=1)
        return self.head(pooled)


class HardSharingMTL(nn.Module):
    """
    Hard-sharing multi-task learning baseline.

    Shares a BiLSTM encoder and uses separate heads for SOH and VDR,
    but without MAE reconstruction or physics-gated loss weighting.
    """

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        enc_out_dim = hidden_dim * 2  # bidirectional

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.soh_head = nn.Sequential(
            nn.Linear(enc_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.vdr_head = nn.Sequential(
            nn.Linear(enc_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        # Mean pooling over time
        pooled = enc_out.mean(dim=1)
        soh_pred = self.soh_head(pooled)
        vdr_pred = self.vdr_head(pooled)
        return soh_pred, vdr_pred

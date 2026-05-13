"""
plot_style.py — Shared publication-quality style for PG-M2TN figures
=====================================================================
Matches the visual language of Nature Machine Intelligence, Nature Energy,
and ICML/NeurIPS proceedings:

  • Helvetica-equivalent sans-serif font (matplotlib: "DejaVu Sans" or system Arial)
  • No top / right spines (clean open-frame axes)
  • Subtle light-gray grid on y-axis only
  • Colorblind-safe palette (Paul Tol "muted" — widely used in Nature papers)
  • Thin, consistent line weights (axes 0.8 pt, data lines 1.8 pt)
  • All legends placed OUTSIDE the plotting area (below or right)
  • High-DPI output (600 dpi for vector-equivalent quality)
  • Panel labels (A), (B) … applied consistently

Usage
-----
    from plot_style import apply_nature_style, PALETTE, add_panel_label, outside_legend
    apply_nature_style()
    fig, axes = plt.subplots(...)
    ...
    outside_legend(fig, axes)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba

# ─────────────────────────────────────────────────────────────────────────────
# Paul Tol "Muted" — 9-color colorblind-safe palette (used by Nature journals)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "blue":        "#332288",   # indigo-blue   (primary / Full model)
    "cyan":        "#88CCEE",   # sky cyan
    "teal":        "#44AA99",   # teal          (secondary / comparisons)
    "green":       "#117733",   # forest green
    "olive":       "#999933",   # olive
    "sand":        "#DDCC77",   # sand yellow
    "rose":        "#CC6677",   # rose          (ablation / negative)
    "wine":        "#882255",   # wine
    "purple":      "#AA4499",   # purple
    "gray":        "#BBBBBB",   # light gray    (baselines)
    "dark_gray":   "#555555",   # dark gray
    "black":       "#000000",
}

# Convenience aliases used across experiments
C_FULL     = PALETTE["blue"]      # PG-M2TN Full
C_NO_MAE   = PALETTE["rose"]      # No-MAE ablation
C_NO_GATE  = PALETTE["teal"]      # No-Gating ablation
C_STATIC   = PALETTE["gray"]      # Static / baseline
C_ACCENT   = PALETTE["sand"]      # Accent / annotation
C_VDR      = PALETTE["wine"]      # VDR weight lines
C_SOH      = PALETTE["blue"]      # SOH lines
C_RECON    = PALETTE["teal"]      # Reconstructed signal
C_ORIG     = PALETTE["dark_gray"] # Original signal
C_MASK     = PALETTE["rose"]      # Masked region highlight
C_ATTN     = PALETTE["sand"]      # Attention overlay


def apply_nature_style(font_scale: float = 1.0) -> None:
    """
    Apply Nature-journal rcParams globally.

    Call this once at the top of every plotting block, before any
    plt.subplots() calls.  font_scale lets you bump sizes up for
    posters / slides without touching the base style.
    """
    base_fs = 8.0 * font_scale       # body text / tick labels
    label_fs = 9.0 * font_scale      # axis labels
    title_fs = 9.5 * font_scale      # subplot titles
    legend_fs = 7.5 * font_scale

    matplotlib.rcParams.update({
        # ── Font ──────────────────────────────────────────────────────────
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size":            base_fs,
        "axes.labelsize":       label_fs,
        "axes.titlesize":       title_fs,
        "xtick.labelsize":      base_fs,
        "ytick.labelsize":      base_fs,
        "legend.fontsize":      legend_fs,
        "figure.titlesize":     10.0 * font_scale,

        # ── Line weights ─────────────────────────────────────────────────
        "lines.linewidth":      1.8,
        "lines.markersize":     5.0,
        "patch.linewidth":      0.8,
        "axes.linewidth":       0.8,
        "grid.linewidth":       0.5,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.width":    0.5,
        "ytick.minor.width":    0.5,
        "xtick.major.size":     3.5,
        "ytick.major.size":     3.5,
        "xtick.minor.size":     2.0,
        "ytick.minor.size":     2.0,

        # ── Axes style ────────────────────────────────────────────────────
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.grid.axis":       "y",     # horizontal guide lines only
        "grid.alpha":           0.25,
        "grid.color":           "#C0C0C0",
        "axes.axisbelow":       True,    # grid behind data
        "axes.facecolor":       "white",
        "figure.facecolor":     "white",

        # ── Ticks ─────────────────────────────────────────────────────────
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.top":            False,
        "ytick.right":          False,

        # ── Legend ────────────────────────────────────────────────────────
        "legend.frameon":       True,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#CCCCCC",
        "legend.borderpad":     0.5,
        "legend.handlelength":  1.8,
        "legend.handletextpad": 0.5,
        "legend.labelspacing":  0.35,

        # ── Saving ────────────────────────────────────────────────────────
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.transparent":  False,

        # ── Color cycle ────────────────────────────────────────────────────
        "axes.prop_cycle": matplotlib.cycler(color=[
            PALETTE["blue"], PALETTE["rose"], PALETTE["teal"],
            PALETTE["sand"], PALETTE["purple"], PALETTE["green"],
            PALETTE["gray"], PALETTE["olive"], PALETTE["wine"],
        ]),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helper: panel label  (A), (B), …
# ─────────────────────────────────────────────────────────────────────────────
def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.06,
                    fontsize: float = 10.0, fontweight: str = "bold") -> None:
    """
    Place a bold panel label '(A)', '(B)' … in the upper-left corner,
    outside the axes frame.  Call after tight_layout / constrained_layout.
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va="bottom", ha="right")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: outside legend (below figure)
# ─────────────────────────────────────────────────────────────────────────────
def outside_legend(fig, handles_or_ax, ncol: int = 3,
                   y_anchor: float = -0.04, fontsize: float = 8.0,
                   title=None) -> None:
    """
    Place a shared legend centred below the figure.

    Parameters
    ----------
    handles_or_ax : list of Artist  OR  Axes
        Pass a list of line handles (preferred) or an Axes whose handles
        will be extracted automatically.
    y_anchor : float
        Vertical position in figure-fraction coordinates (negative = below).
    """
    if hasattr(handles_or_ax, "get_legend_handles_labels"):
        handles, labels = handles_or_ax.get_legend_handles_labels()
    else:
        handles = handles_or_ax
        labels = [h.get_label() for h in handles]

    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y_anchor),
              ncol=ncol, fontsize=fontsize, frameon=True,
              framealpha=0.92, edgecolor="#CCCCCC",
              borderpad=0.6, handlelength=2.0, handletextpad=0.5)
    if title:
        kw["title"] = title
    fig.legend(handles, labels, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: strip twin-axis right spine for dual-axis plots
# ─────────────────────────────────────────────────────────────────────────────
def style_twin_ax(ax2, color: str = PALETTE["wine"]) -> None:
    """
    Apply Nature-consistent style to a twinx() axis:
    thin right spine, matching tick color.
    """
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["right"].set_color(color)
    ax2.tick_params(axis="y", colors=color, width=0.8, length=3.5)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.label.set_color(color)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: annotate improvement % between two bar groups
# ─────────────────────────────────────────────────────────────────────────────
def annotate_improvement(ax, x_center: float, val_base: float, val_ours: float,
                          y_offset_frac: float = 0.04) -> None:
    """
    Draw a '+X.X%' / '-X.X%' annotation above a bar pair.
    Positive (improvement) shown in PALETTE["blue"], negative in rose.
    """
    if abs(val_base) < 1e-9:
        return
    pct = (val_base - val_ours) / abs(val_base) * 100.0
    color = PALETTE["blue"] if pct > 0 else PALETTE["rose"]
    y_top = max(val_base, val_ours)
    ax.annotate(f"{pct:+.1f}%",
                xy=(x_center, y_top * (1.0 + y_offset_frac)),
                fontsize=7.5, ha="center", va="bottom",
                color=color, fontweight="bold")

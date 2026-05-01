"""
plots_pca.py — Графіки для МГК (пункт 5, Лаб. 5)

Функції:
  pca_panel_figure   — 2×2: кореляційне поле, 2D гістограма,
                       центр розсіювання, кореляційна матриця
  pca_variance_figure — Scree plot + накопичена дисперсія
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker

from plots import (
    _style_fig, _style_ax,
    BG, PANEL, TEXT, GRID, ACCENT, ACCENT2, GREEN, YELLOW, CORR_CMAP,
)


# ──────────────────────────────────────────────────────────────
# ДОПОМІЖНА: теплова карта кореляційної матриці в осі ax
# ──────────────────────────────────────────────────────────────
def _corr_heatmap_ax(ax, fig, R: np.ndarray, labels: list, title: str):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

    p  = len(labels)
    im = ax.imshow(R, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.tick_params(colors=TEXT, labelsize=6)
    cbar.outline.set_edgecolor(GRID)

    ax.set_xticks(range(p))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8, color=TEXT)
    ax.set_yticks(range(p))
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.tick_params(colors=TEXT)

    for i in range(p):
        for j in range(p):
            val = R[i, j]
            fg  = 'white' if abs(val) > 0.45 else TEXT
            ax.text(j, i, f"{val:.3f}",
                    ha='center', va='center',
                    fontsize=8, color=fg, fontweight='bold')

    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=6)


# ──────────────────────────────────────────────────────────────
# ОСНОВНА ПАНЕЛЬ: 4 графіки в сітці 2×2
# ──────────────────────────────────────────────────────────────
def pca_panel_figure(data: np.ndarray,
                     col_names: list,
                     center: np.ndarray,
                     corr_matrix: np.ndarray,
                     title: str) -> Figure:
    """
    Параметри
    ----------
    data        : масив (n × k) — дані в поточній системі координат
    col_names   : назви осей (усі k)
    center      : вектор центрів (усі k)
    corr_matrix : кореляційна матриця (k × k)
    title       : заголовок фігури

    Для 2D-графіків (scatter, hist2d) використовуються перші 2 виміри.
    Центр розсіювання та кор. матриця — усі виміри.
    """
    fig = Figure(figsize=(12, 8), dpi=110)
    _style_fig(fig)
    fig.suptitle(title, fontsize=12, fontweight='bold', color=TEXT, y=1.005)

    ax_scatter = fig.add_subplot(2, 2, 1)
    ax_hist2d  = fig.add_subplot(2, 2, 2)
    ax_center  = fig.add_subplot(2, 2, 3)
    ax_corr    = fig.add_subplot(2, 2, 4)

    # Перші два виміри для 2D-графіків
    cx_name = col_names[0]
    cy_name = col_names[1] if len(col_names) > 1 else col_names[0]
    x = data[:, 0]
    y = data[:, 1] if data.shape[1] > 1 else data[:, 0]
    cx = float(center[0])
    cy = float(center[1]) if len(center) > 1 else float(center[0])

    # ── 1. Кореляційне поле ───────────────────────────────────
    _style_ax(ax_scatter,
              f"Кореляційне поле: {cx_name} vs {cy_name}",
              cx_name, cy_name)
    ax_scatter.scatter(x, y,
                       s=16, color=ACCENT, alpha=0.60,
                       edgecolors='none', rasterized=True)
    # Центр розсіювання
    ax_scatter.scatter(cx, cy,
                       marker='+', s=260, color=ACCENT2,
                       linewidths=2.5, zorder=10, label=f"Центр ({cx:.3f}, {cy:.3f})")
    ax_scatter.axvline(cx, color=ACCENT2, lw=0.8, ls='--', alpha=0.55)
    ax_scatter.axhline(cy, color=ACCENT2, lw=0.8, ls='--', alpha=0.55)
    ax_scatter.legend(facecolor=PANEL, edgecolor=GRID,
                      labelcolor=TEXT, fontsize=7, loc='upper right')

    # ── 2. 2D гістограма ─────────────────────────────────────
    _style_ax(ax_hist2d, f"2D гістограма: {cx_name} × {cy_name}",
              cx_name, cy_name)
    bins = max(8, int(np.sqrt(len(x))))
    _h, _xe, _ye, img = ax_hist2d.hist2d(
        x, y, bins=bins, cmap='Blues',
        density=False)
    cbar2 = fig.colorbar(img, ax=ax_hist2d, shrink=0.85, pad=0.03)
    cbar2.ax.tick_params(colors=TEXT, labelsize=6)
    cbar2.outline.set_edgecolor(GRID)
    ax_hist2d.set_facecolor(PANEL)
    ax_hist2d.tick_params(colors=TEXT, labelsize=7)
    for sp in ax_hist2d.spines.values():
        sp.set_edgecolor(GRID)

    # ── 3. Центр розсіювання (текст) ─────────────────────────
    ax_center.set_facecolor(PANEL)
    ax_center.set_xticks([])
    ax_center.set_yticks([])
    for sp in ax_center.spines.values():
        sp.set_edgecolor(GRID)
    ax_center.set_title("Центр розсіювання (μ)",
                         color=TEXT, fontsize=10, fontweight='bold', pad=6)

    lines = [f"  {col_names[i]:>8}:  {float(center[i]):+.5f}"
             for i in range(len(col_names))]
    block = "\n".join(lines)
    ax_center.text(0.5, 0.5, block,
                   transform=ax_center.transAxes,
                   ha='center', va='center',
                   fontsize=10, color=YELLOW,
                   fontfamily='Consolas',
                   bbox=dict(facecolor=BG, edgecolor=GRID,
                             boxstyle='round,pad=0.7', linewidth=1))

    # ── 4. Кореляційна матриця ───────────────────────────────
    _corr_heatmap_ax(ax_corr, fig, corr_matrix, col_names,
                     "Кореляційна матриця")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# SCREE PLOT + накопичена дисперсія
# ──────────────────────────────────────────────────────────────
def pca_variance_figure(eigenvalues: np.ndarray,
                         var_pct: np.ndarray,
                         cum_pct: np.ndarray) -> Figure:
    """Scree plot (ліворуч) + стовпці % дисперсії з накопиченою кривою."""
    p   = len(eigenvalues)
    fig = Figure(figsize=(10, 4.5), dpi=110)
    _style_fig(fig)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x   = np.arange(1, p + 1)

    # Scree plot
    _style_ax(ax1, "Scree plot (Власні числа)", "Головна компонента", "λ")
    ax1.bar(x, eigenvalues, color=ACCENT, alpha=0.80, edgecolor=BG, linewidth=0.5)
    ax1.plot(x, eigenvalues, 'o-', color=GREEN, lw=1.8, markersize=6,
             markerfacecolor=GREEN, markeredgecolor=BG)
    ax1.axhline(1.0, color=ACCENT2, ls='--', lw=1.2, alpha=0.85, label='λ = 1 (критерій Кайзера)')
    ax1.set_xticks(x)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    for i, (xi, lam) in enumerate(zip(x, eigenvalues)):
        ax1.text(xi, lam + 0.03, f"{lam:.3f}",
                 ha='center', fontsize=7, color=YELLOW)

    # % дисперсії + накопичена крива
    _style_ax(ax2, "Пояснена дисперсія", "Головна компонента", "%")
    ax2.bar(x, var_pct, color=ACCENT, alpha=0.75, edgecolor=BG, linewidth=0.5,
            label='% дисперсії')
    ax2.plot(x, cum_pct, 'o-', color=ACCENT2, lw=1.8, markersize=6,
             markerfacecolor=ACCENT2, markeredgecolor=BG, label='Накопичений %')
    ax2.axhline(80, color=YELLOW, ls='--', lw=1.0, alpha=0.7, label='80%')
    ax2.set_ylim(0, 115)
    ax2.set_xticks(x)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    for i, (xi, pct, cum) in enumerate(zip(x, var_pct, cum_pct)):
        ax2.text(xi, pct + 1.5, f"{pct:.1f}%",
                 ha='center', fontsize=7, color=YELLOW)

    fig.tight_layout()
    return fig
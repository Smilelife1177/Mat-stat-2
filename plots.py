"""
plots.py — Всі графіки для Лаб. 5 (matplotlib, вбудовуються в tkinter)
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


# ── Кольорова схема (темна, академічна) ──────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#4f8ef7"
ACCENT2 = "#e05c7a"
TEXT    = "#d6dae8"
GRID    = "#252836"
GREEN   = "#4ecb88"
YELLOW  = "#f5c542"

# Власна дивергентна colormap
_cmap_colors = [(0.87, 0.22, 0.38), (0.12, 0.13, 0.20), (0.31, 0.56, 0.97)]
CORR_CMAP = LinearSegmentedColormap.from_list("corr", _cmap_colors, N=256)


def _style_fig(fig):
    fig.patch.set_facecolor(BG)


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)
    if title:  ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


# ──────────────────────────────────────────────────────────────
# HEATMAP (парні / часткові)
# ──────────────────────────────────────────────────────────────
def heatmap_figure(matrix: np.ndarray, cols: list,
                   title: str = "Матриця кореляцій",
                   annot_size: int = 9) -> Figure:
    p   = len(cols)
    fig = Figure(figsize=(max(5, p * 1.1), max(4, p * 1.0)), dpi=110)
    _style_fig(fig)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(PANEL)

    im = ax.imshow(matrix, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.outline.set_edgecolor(GRID)

    ax.set_xticks(range(p))
    ax.set_xticklabels(cols, rotation=40, ha='right', fontsize=8, color=TEXT)
    ax.set_yticks(range(p))
    ax.set_yticklabels(cols, fontsize=8, color=TEXT)
    ax.spines[:].set_color(GRID)

    for i in range(p):
        for j in range(p):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.45 else TEXT
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=annot_size, color=color, fontweight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# SCATTER MATRIX
# ──────────────────────────────────────────────────────────────
def scatter_matrix_figure(df) -> Figure:
    import pandas as pd
    cols = list(df.columns)
    p    = len(cols)
    sz   = max(3.2, p * 1.8)
    fig  = Figure(figsize=(sz, sz), dpi=100)
    _style_fig(fig)

    for i in range(p):
        for j in range(p):
            idx = i * p + j + 1
            ax  = fig.add_subplot(p, p, idx)
            ax.set_facecolor(PANEL)
            ax.spines[:].set_color(GRID)
            ax.tick_params(colors=TEXT, labelsize=6)

            if i == j:
                ax.hist(df.iloc[:, i], bins=14, color=ACCENT,
                        alpha=0.8, edgecolor=BG, linewidth=0.4)
                ax.set_title(cols[i], fontsize=8, color=ACCENT, fontweight='bold', pad=3)
            else:
                r, _ = stats.pearsonr(df.iloc[:, j], df.iloc[:, i])
                c    = ACCENT if r >= 0 else ACCENT2
                ax.scatter(df.iloc[:, j], df.iloc[:, i],
                           s=6, alpha=0.45, color=c, linewidths=0)
                ax.text(0.05, 0.88, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=7, color=YELLOW, fontweight='bold')

            if i == p - 1: ax.set_xlabel(cols[j], fontsize=7, color=TEXT)
            if j == 0:     ax.set_ylabel(cols[i], fontsize=7, color=TEXT)
            ax.grid(True, color=GRID, linewidth=0.4, linestyle='--', alpha=0.6)

    fig.suptitle("Матриця діаграм розкиду", fontsize=12,
                 fontweight='bold', color=TEXT, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# MULTIPLE CORRELATION BAR
# ──────────────────────────────────────────────────────────────
def multiple_bar_figure(rows: list) -> Figure:
    cols   = [r['Змінна'] for r in rows]
    Rmults = [r['R_mult'] for r in rows]
    R2s    = [r['R²']     for r in rows]
    sigs   = [r['Значущий'] for r in rows]

    fig = Figure(figsize=(max(6, len(cols) * 1.0), 4.5), dpi=110)
    _style_fig(fig)
    ax  = fig.add_subplot(111)
    _style_ax(ax, "Множинні коефіцієнти кореляції", "Змінна", "Значення")

    x = np.arange(len(cols))
    w = 0.35
    bars1 = ax.bar(x - w/2, Rmults, w, color=ACCENT,   alpha=0.85,
                   label='R_mult', edgecolor=BG, linewidth=0.5)
    bars2 = ax.bar(x + w/2, R2s,    w, color=ACCENT2,  alpha=0.85,
                   label='R²',     edgecolor=BG, linewidth=0.5)

    ax.axhline(0.5, color=GREEN, linestyle='--', linewidth=1, alpha=0.7, label='R=0.5')
    ax.set_xticks(x)
    ax.set_xticklabels(cols, color=TEXT, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    for bar, val, sig in zip(bars1, Rmults, sigs):
        c = GREEN if "Так" in sig else ACCENT2
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontsize=8, color=c, fontweight='bold')

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# ОПТИМАЛЬНА КІЛЬКІСТЬ КЛАСІВ (формула з посібника)
# ──────────────────────────────────────────────────────────────
def optimal_bins(n: int) -> int:
    """
    Формула з посібника:
      При N < 100:   M = floor(sqrt(N)),   якщо непарне
                     M = floor(sqrt(N))-1, якщо парне
      При N >= 100:  M = floor(cbrt(N)),   якщо непарне
                     M = floor(cbrt(N))-1, якщо парне
    """
    if n < 100:
        m = int(np.floor(np.sqrt(n)))
    else:
        m = int(np.floor(np.cbrt(n)))   # кубічний корінь = N^(1/3)

    # Якщо парне — зменшуємо на 1, щоб зробити непарним (бажано непарне)
    if m % 2 == 0:
        m -= 1

    return max(m, 3)   # мінімум 3 класи


# ──────────────────────────────────────────────────────────────
# HISTOGRAMS (первинний аналіз)
# ──────────────────────────────────────────────────────────────
def histograms_figure(df) -> Figure:
    cols  = list(df.columns)
    p     = len(cols)
    ncols = min(p, 3)
    nrows = (p + ncols - 1) // ncols
    fig   = Figure(figsize=(ncols * 3.5, nrows * 3.0), dpi=100)
    _style_fig(fig)

    for idx, col in enumerate(cols):
        ax   = fig.add_subplot(nrows, ncols, idx + 1)
        data = df[col].dropna().values
        n    = len(data)
        M    = optimal_bins(n)

        _style_ax(ax, title=col, xlabel="Значення", ylabel="Частота")

        ax.hist(data, bins=M, color=ACCENT, alpha=0.8,
                edgecolor=BG, linewidth=0.4)

        # Підпис кількості класів (формула посібника)
        root_sym = 'sqrt(N)' if n < 100 else 'cbrt(N)'
        root_val = float(np.sqrt(n)) if n < 100 else float(n ** (1/3))
        formula  = 'M=%d  (%s=%.2f, N=%d)' % (M, root_sym, root_val, n)
        ax.text(0.97, 0.97, formula, transform=ax.transAxes,
                fontsize=6.5, color=YELLOW, ha='right', va='top',
                bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle='round,pad=0.3'))

        # Крива нормального розподілу
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 300)
        pdf = stats.norm.pdf(x, mu, sigma)
        h   = (data.max() - data.min()) / M        # ширина класу
        count_scale = n * h
        ax.plot(x, pdf * count_scale, color=ACCENT2, linewidth=1.5, label='Норм. розп.')
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    fig.suptitle("Гістограми ознак", fontsize=12, fontweight='bold', color=TEXT)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# PARTIAL CI PLOT
# ──────────────────────────────────────────────────────────────
def partial_ci_figure(pairs: list) -> Figure:
    """Графік часткових кореляцій з довірчими інтервалами (forest plot)."""
    valid = [p for p in pairs if p['ДІ нижня'] != "—"]
    if not valid:
        # Просто barplot
        valid = pairs

    labels = [p['Пара'] for p in valid]
    rvals  = [p['r_part'] for p in valid]
    sigs   = ["✔" in p['Значущий'] for p in valid]

    fig = Figure(figsize=(7, max(3.5, len(labels) * 0.55)), dpi=110)
    _style_fig(fig)
    ax  = fig.add_subplot(111)
    _style_ax(ax, "Часткові коефіцієнти кореляції (forest plot)",
              "r часткова", "Пара змінних")

    y = np.arange(len(labels))
    for i, (row, sig) in enumerate(zip(valid, sigs)):
        r   = row['r_part']
        c   = GREEN if sig else ACCENT2
        ax.scatter(r, i, color=c, s=60, zorder=5)
        if row['ДІ нижня'] != "—":
            ax.hlines(i, row['ДІ нижня'], row['ДІ верхня'],
                      color=c, linewidth=2.5, alpha=0.6, zorder=4)

    ax.axvline(0, color=TEXT, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.set_xlim(-1.1, 1.1)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN,  markersize=7, label='Значущий'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT2, markersize=7, label='Незначущий'),
    ]
    ax.legend(handles=legend_elements, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=8)

    fig.tight_layout()
    return fig
"""
plots_pca.py — Графічні візуалізації для МГК (PCA).

Функції:
  1. pca_scatter_matrix_figure   — Матриця діаграм розкиду (початкові дані, ГК >95%, відновлені дані)
  2. pca_variance_figure          — Scree plot + накопичена дисперсія (з порогом 95% та критерієм Кайзера)
  3. pca_3d_figure                — 3D-реалізація в просторі головних компонент (PC1, PC2, PC3)
  4. pca_reconstruction_figure    — Діагностичні графіки точності зворотного перетворення (X vs X̂)
  5. pca_panel_figure             — 2D-зріз (кореляційне поле, 2D-гістограма, центр, кор. матриця)
"""

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

from plots import (
    _style_fig, _style_ax,
    BG, PANEL, TEXT, GRID, ACCENT, ACCENT2, GREEN, YELLOW, CORR_CMAP,
)


# ──────────────────────────────────────────────────────────────
# 1. МАТРИЦЯ ДІАГРАМ РОЗКИДУ (SCATTER MATRIX)
# ──────────────────────────────────────────────────────────────
def pca_scatter_matrix_figure(df: pd.DataFrame,
                             title: str = "Матриця діаграм розкиду",
                             subtitle: str = "",
                             color: str = ACCENT) -> Figure:
    """
    Універсальна матриця діаграм розкиду для темної теми.
    - Діагональ: гістограми розподілу
    - Нижній трикутник: діаграми розкиду (точки)
    - Верхній трикутник: парні коефіцієнти кореляції Пірсона (Corr)
    """
    cols = list(df.columns)
    p = len(cols)
    if p == 0:
        fig = Figure(figsize=(6, 4), dpi=100)
        _style_fig(fig)
        return fig

    # Розрахунок розмірів сітки
    cell = max(2.0, min(3.0, 9.6 / p))
    sz = cell * p
    fig = Figure(figsize=(sz + 0.8, sz + 0.9), dpi=105)
    _style_fig(fig)

    # Заголовок та підзаголовок
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', color=TEXT, y=0.985)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, fontsize=8.5, color=YELLOW, ha='center')

    margin_l = 0.08
    margin_b = 0.08
    margin_r = 0.04
    margin_t = 0.09 if subtitle else (0.07 if title else 0.05)
    usable_w = 1.0 - margin_l - margin_r
    usable_h = 1.0 - margin_b - margin_t
    cell_w = usable_w / p
    cell_h = usable_h / p

    # Попереднє обчислення кореляцій
    R = {}
    for i in range(p):
        for j in range(p):
            if i != j:
                s_i = df.iloc[:, i].dropna().values
                s_j = df.iloc[:, j].dropna().values
                if len(s_i) > 1 and np.std(s_i) > 1e-12 and np.std(s_j) > 1e-12:
                    r, _ = stats.pearsonr(s_i, s_j)
                else:
                    r = 0.0
                R[(i, j)] = r

    for i in range(p):        # рядок
        for j in range(p):    # стовпець
            left = margin_l + j * cell_w
            bottom = margin_b + (p - 1 - i) * cell_h
            ax = fig.add_axes([left, bottom, cell_w * 0.94, cell_h * 0.94])
            ax.set_facecolor(PANEL)

            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
                sp.set_linewidth(0.6)

            ax.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.65)
            ax.set_axisbelow(True)
            ax.tick_params(colors=TEXT, labelsize=7, length=3, width=0.5)

            # Підписи тільки на зовнішніх краях
            if j != 0:
                ax.tick_params(labelleft=False)
            if i != p - 1:
                ax.tick_params(labelbottom=False)

            # ── ДІАГОНАЛЬ: гістограма ──────────────────────────
            if i == j:
                data = df.iloc[:, i].dropna().values
                bins = max(8, int(np.sqrt(len(data))))
                ax.hist(data, bins=bins, color=color,
                        edgecolor=BG, linewidth=0.4, alpha=0.85)
                # Назва ознаки зверху
                ax.set_title(cols[i], fontsize=8, color=YELLOW,
                             fontweight='bold', pad=3)

            # ── НИЖНІЙ ТРИКУТНИК: scatter plot ────────────────
            elif i > j:
                ax.scatter(df.iloc[:, j], df.iloc[:, i],
                           s=8, color=color, alpha=0.55,
                           edgecolors='none', rasterized=True)

            # ── ВЕРХНІЙ ТРИКУТНИК: значення кореляції ─────────
            else:
                r = R.get((i, j), 0.0)
                ax.set_xticks([])
                ax.set_yticks([])

                # Колір та розмір залежно від величини |r|
                abs_r = abs(r)
                if abs_r >= 0.70:
                    r_color = ACCENT2
                    fs = 11
                elif abs_r >= 0.35:
                    r_color = GREEN
                    fs = 10
                elif abs_r >= 0.10:
                    r_color = TEXT
                    fs = 9
                else:
                    # Близько до 0 (незалежні ГК)
                    r_color = "#8f96a3"
                    fs = 9

                ax.text(0.5, 0.60, 'Corr:',
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=7, color='#7e8597')
                ax.text(0.5, 0.35, f'{r:+.3f}' if abs_r >= 0.0005 else '0.000',
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=fs, color=r_color,
                        fontweight='bold')

    return fig


# ──────────────────────────────────────────────────────────────
# 2. SCREE PLOT ТА ПОЯСНЕНА ДИСПЕРСІЯ
# ──────────────────────────────────────────────────────────────
def pca_variance_figure(eigenvalues: np.ndarray,
                         var_pct: np.ndarray,
                         cum_pct: np.ndarray,
                         threshold: float = 95.0,
                         k_retained: int = None) -> Figure:
    """Scree plot (власні числа) + графік накопиченої дисперсії з порогом 95%."""
    p   = len(eigenvalues)
    fig = Figure(figsize=(10.5, 4.8), dpi=105)
    _style_fig(fig)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x   = np.arange(1, p + 1)
    pc_labels = [f"PC{i}" for i in x]

    # ── 1. Scree plot ─────────────────────────────────────────
    _style_ax(ax1, "Scree plot (Власні числа λ)", "Головна компонента", "Власне число (λ)")
    bar_colors = [GREEN if (k_retained and i <= k_retained) else ACCENT for i in x]
    ax1.bar(x, eigenvalues, color=bar_colors, alpha=0.78, edgecolor=BG, linewidth=0.6,
            label='λ головної компоненти')
    ax1.plot(x, eigenvalues, 'o-', color=YELLOW, lw=1.8, markersize=6,
             markerfacecolor=YELLOW, markeredgecolor=BG)

    # Критерій Кайзера (λ = 1)
    ax1.axhline(1.0, color=ACCENT2, ls='--', lw=1.3, alpha=0.9,
                label='λ = 1 (критерій Кайзера)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pc_labels, fontsize=8)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, loc='upper right')

    for xi, lam in zip(x, eigenvalues):
        ax1.text(xi, lam + max(eigenvalues) * 0.025, f"{lam:.3f}",
                 ha='center', fontsize=7.5, color=YELLOW, fontweight='bold')

    # ── 2. % дисперсії + накопичена крива ──────────────────────
    _style_ax(ax2, "Пояснена та накопичена дисперсія", "Головна компонента", "% від загальної дисперсії")
    ax2.bar(x, var_pct, color=ACCENT, alpha=0.68, edgecolor=BG, linewidth=0.6,
            label='% дисперсії на ГК')
    ax2.plot(x, cum_pct, 's-', color=GREEN, lw=2.0, markersize=6,
             markerfacecolor=GREEN, markeredgecolor=BG, label='Накопичена дисперсія')

    # Поріг відбору (95%)
    ax2.axhline(threshold, color=ACCENT2, ls='--', lw=1.3, alpha=0.9,
                label=f'Поріг {threshold:.0f}% (правило відбору)')

    # Позначка відібраних компонент
    if k_retained is not None and 1 <= k_retained <= p:
        ax2.axvline(k_retained + 0.45, color=YELLOW, ls=':', lw=1.2, alpha=0.85,
                    label=f'Відібрано k={k_retained} ГК ({cum_pct[k_retained-1]:.1f}%)')

    ax2.set_ylim(0, 118)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pc_labels, fontsize=8)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, loc='center right')

    for xi, pct, cum in zip(x, var_pct, cum_pct):
        ax2.text(xi, pct + 2.0, f"{pct:.1f}%",
                 ha='center', fontsize=7.5, color=TEXT)
        ax2.text(xi, cum + 2.0, f"{cum:.1f}%",
                 ha='center', fontsize=7.5, color=GREEN, fontweight='bold')

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 3. 3D-РЕАЛІЗАЦІЯ В ПРОСТОРІ ГОЛОВНИХ КОМПОНЕНТ
# ──────────────────────────────────────────────────────────────
def pca_3d_figure(Z: np.ndarray,
                  var_pct: np.ndarray,
                  pc_names: list = None,
                  title: str = "3D-реалізація в просторі головних компонент",
                  elev: float = 24.0,
                  azim: float = 46.0) -> Figure:
    """
    3D-візуалізація спостережень у просторі трьох перших головних компонент (PC1, PC2, PC3).
    Підтримує обертання мишею через інтерактивний тулбар.
    """
    fig = Figure(figsize=(9.2, 7.0), dpi=105)
    _style_fig(fig)

    n, p = Z.shape
    if p < 3:
        ax = fig.add_subplot(111)
        _style_ax(ax, title)
        ax.text(0.5, 0.5,
                f"Для повноцінної 3D-візуалізації потрібно щонайменше 3 ознаки.\n"
                f"Поточна розмірність: {p} ознак(и).",
                ha='center', va='center', color=YELLOW, fontsize=11,
                transform=ax.transAxes)
        return fig

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)

    # Стилізація граней 3D куба
    pane_color = (0.10, 0.11, 0.16, 1.0)
    ax.xaxis.pane.set_facecolor(pane_color)
    ax.yaxis.pane.set_facecolor(pane_color)
    ax.zaxis.pane.set_facecolor(pane_color)

    edge_color = (0.16, 0.18, 0.25, 1.0)
    ax.xaxis.pane.set_edgecolor(edge_color)
    ax.yaxis.pane.set_edgecolor(edge_color)
    ax.zaxis.pane.set_edgecolor(edge_color)

    ax.tick_params(colors=TEXT, labelsize=7.5, pad=2)
    ax.grid(True, color=GRID, linestyle=':', alpha=0.6)

    # Дані 3D
    x = Z[:, 0]
    y = Z[:, 1]
    z = Z[:, 2]

    # Кольорове кодування за PC1 або відстанню від центру
    dist = np.sqrt(x**2 + y**2 + z**2)
    scatter = ax.scatter(x, y, z, c=x, cmap='plasma', s=22, alpha=0.75,
                         edgecolors='none', depthshade=True)

    # Центр розсіювання (початок координат у просторі МГК: 0, 0, 0)
    ax.scatter([0], [0], [0], color=ACCENT2, marker='+', s=260,
               linewidths=2.8, zorder=20, label='Центр (0, 0, 0)')

    # Осі через початок координат (пунктирні лінії)
    pad_x = max(abs(x.min()), abs(x.max())) * 1.15
    pad_y = max(abs(y.min()), abs(y.max())) * 1.15
    pad_z = max(abs(z.min()), abs(z.max())) * 1.15
    ax.plot([-pad_x, pad_x], [0, 0], [0, 0], color=GRID, ls='--', lw=0.9, alpha=0.8)
    ax.plot([0, 0], [-pad_y, pad_y], [0, 0], color=GRID, ls='--', lw=0.9, alpha=0.8)
    ax.plot([0, 0], [0, 0], [-pad_z, pad_z], color=GRID, ls='--', lw=0.9, alpha=0.8)

    # Назви осей з часткою дисперсії
    name1 = pc_names[0] if pc_names and len(pc_names) > 0 else "PC1"
    name2 = pc_names[1] if pc_names and len(pc_names) > 1 else "PC2"
    name3 = pc_names[2] if pc_names and len(pc_names) > 2 else "PC3"

    pct1 = var_pct[0] if len(var_pct) > 0 else 0.0
    pct2 = var_pct[1] if len(var_pct) > 1 else 0.0
    pct3 = var_pct[2] if len(var_pct) > 2 else 0.0
    cum3 = pct1 + pct2 + pct3

    ax.set_xlabel(f"{name1} ({pct1:.1f}%)", color=TEXT, fontsize=8.5, labelpad=7)
    ax.set_ylabel(f"{name2} ({pct2:.1f}%)", color=TEXT, fontsize=8.5, labelpad=7)
    ax.set_zlabel(f"{name3} ({pct3:.1f}%)", color=TEXT, fontsize=8.5, labelpad=7)

    ax.set_title(f"{title}\nСумарно 3 ГК пояснюють {cum3:.1f}% дисперсії  (n={n} спостережень)",
                 color=TEXT, fontsize=10.5, fontweight='bold', pad=12)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.60, pad=0.08, aspect=18)
    cbar.ax.tick_params(colors=TEXT, labelsize=6.5)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label(f"Значення координати {name1}", color=TEXT, fontsize=7.5)

    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, loc='upper right')
    ax.view_init(elev=elev, azim=azim)

    # Зберігаємо ax для динамічного обертання через кнопки
    fig._pca_3d_ax = ax
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 4. ДІАГНОСТИКА ЗВОРОТНОГО ПЕРЕТВОРЕННЯ (X vs X̂)
# ──────────────────────────────────────────────────────────────
def pca_reconstruction_figure(X_orig: np.ndarray,
                              X_rec: np.ndarray,
                              cols: list,
                              k_retained: int = 3,
                              retained_var: float = 99.5) -> Figure:
    """
    Поточкове порівняння оригінальних даних X та відновлених X̂ для кожної ознаки.
    Ідеальне відновлення відповідає діагоналі y = x.
    """
    p = len(cols)
    nrows = int(np.ceil(p / 2))
    ncols = 2 if p > 1 else 1

    fig = Figure(figsize=(10.5, max(5.0, nrows * 3.3)), dpi=105)
    _style_fig(fig)
    fig.suptitle(
        f"Зворотне перетворення: Оригінальні дані vs Відновлені дані\n"
        f"Відновлено з k={k_retained} ГК (пояснюють {retained_var:.2f}% дисперсії)",
        color=TEXT, fontsize=11, fontweight='bold', y=0.985)

    for j in range(p):
        ax = fig.add_subplot(nrows, ncols, j + 1)
        orig = X_orig[:, j]
        rec  = X_rec[:, j]
        col_name = cols[j]

        mse = float(np.mean((orig - rec) ** 2))
        std_orig = float(np.std(orig))
        if std_orig > 1e-12:
            r_val, _ = stats.pearsonr(orig, rec)
        else:
            r_val = 1.0

        ss_tot = float(np.sum((orig - np.mean(orig)) ** 2))
        ss_res = float(np.sum((orig - rec) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        _style_ax(ax,
                  f"{col_name}: r = {r_val:.4f}  |  R² = {r2:.4f}  |  MSE = {mse:.5f}",
                  f"Оригінал {col_name}",
                  f"Відновлено {col_name}")

        ax.scatter(orig, rec, s=12, color=ACCENT, alpha=0.55,
                   edgecolors='none', rasterized=True, label='Спостереження')

        # Ідеальна діагональ y = x
        mn = min(float(orig.min()), float(rec.min()))
        mx = max(float(orig.max()), float(rec.max()))
        pad = (mx - mn) * 0.05
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                color=ACCENT2, linestyle='--', lw=1.5, label='Ідеальна відповідність (y = x)')

        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7, loc='upper left')

    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    return fig


# ──────────────────────────────────────────────────────────────
# 5. ДВОМІРНИЙ ЗРІЗ (КОРЕЛЯЦІЙНЕ ПОЛЕ, 2D-ГІСТ, ЦЕНТР, МАТРИЦЯ)
# ──────────────────────────────────────────────────────────────
def pca_panel_figure(data: np.ndarray,
                     col_names: list,
                     center: np.ndarray,
                     corr_matrix: np.ndarray,
                     title: str) -> Figure:
    """
    Панель 2×2: кореляційне поле, 2D гістограма, центр розсіювання, кореляційна матриця.
    """
    fig = Figure(figsize=(11.5, 7.8), dpi=105)
    _style_fig(fig)
    fig.suptitle(title, fontsize=11.5, fontweight='bold', color=TEXT, y=0.99)

    ax_scatter = fig.add_subplot(2, 2, 1)
    ax_hist2d  = fig.add_subplot(2, 2, 2)
    ax_center  = fig.add_subplot(2, 2, 3)
    ax_corr    = fig.add_subplot(2, 2, 4)

    cx_name = col_names[0]
    cy_name = col_names[1] if len(col_names) > 1 else col_names[0]
    x = data[:, 0]
    y = data[:, 1] if data.shape[1] > 1 else data[:, 0]
    cx = float(center[0])
    cy = float(center[1]) if len(center) > 1 else float(center[0])

    # 1. Кореляційне поле
    _style_ax(ax_scatter, f"Кореляційне поле: {cx_name} vs {cy_name}", cx_name, cy_name)
    ax_scatter.scatter(x, y, s=16, color=ACCENT, alpha=0.60, edgecolors='none', rasterized=True)
    ax_scatter.scatter(cx, cy, marker='+', s=240, color=ACCENT2, linewidths=2.5,
                       zorder=10, label=f"Центр ({cx:.3f}, {cy:.3f})")
    ax_scatter.axvline(cx, color=ACCENT2, lw=0.8, ls='--', alpha=0.55)
    ax_scatter.axhline(cy, color=ACCENT2, lw=0.8, ls='--', alpha=0.55)
    ax_scatter.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, loc='upper right')

    # 2. 2D гістограма
    _style_ax(ax_hist2d, f"2D гістограма: {cx_name} × {cy_name}", cx_name, cy_name)
    bins = max(8, int(np.sqrt(len(x))))
    _h, _xe, _ye, img = ax_hist2d.hist2d(x, y, bins=bins, cmap='Blues', density=False)
    cbar2 = fig.colorbar(img, ax=ax_hist2d, shrink=0.85, pad=0.03)
    cbar2.ax.tick_params(colors=TEXT, labelsize=6.5)
    cbar2.outline.set_edgecolor(GRID)
    ax_hist2d.set_facecolor(PANEL)
    ax_hist2d.tick_params(colors=TEXT, labelsize=7)
    for sp in ax_hist2d.spines.values():
        sp.set_edgecolor(GRID)

    # 3. Центр розсіювання
    ax_center.set_facecolor(PANEL)
    ax_center.set_xticks([])
    ax_center.set_yticks([])
    for sp in ax_center.spines.values():
        sp.set_edgecolor(GRID)
    ax_center.set_title("Центр розсіювання (вектор середніх μ)",
                        color=TEXT, fontsize=9.5, fontweight='bold', pad=6)

    lines = [f"  {col_names[i]:>12}:  {float(center[i]):+.5f}"
             for i in range(len(col_names))]
    block = "\n".join(lines)
    ax_center.text(0.5, 0.5, block,
                   transform=ax_center.transAxes,
                   ha='center', va='center',
                   fontsize=9.5, color=YELLOW,
                   fontfamily='Consolas',
                   bbox=dict(facecolor=BG, edgecolor=GRID,
                             boxstyle='round,pad=0.7', linewidth=1))

    # 4. Кореляційна матриця
    _corr_heatmap_ax(ax_corr, fig, corr_matrix, col_names, "Кореляційна матриця")

    fig.tight_layout()
    return fig


def _corr_heatmap_ax(ax, fig, R: np.ndarray, labels: list, title: str):
    """Допоміжна функція: теплова карта кореляційної матриці."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

    p  = len(labels)
    im = ax.imshow(R, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.tick_params(colors=TEXT, labelsize=6.5)
    cbar.outline.set_edgecolor(GRID)

    ax.set_xticks(range(p))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7.5, color=TEXT)
    ax.set_yticks(range(p))
    ax.set_yticklabels(labels, fontsize=7.5, color=TEXT)
    ax.tick_params(colors=TEXT)

    for i in range(p):
        for j in range(p):
            val = R[i, j]
            fg  = 'white' if abs(val) > 0.45 else TEXT
            ax.text(j, i, f"{val:.3f}",
                    ha='center', va='center',
                    fontsize=7.5, color=fg, fontweight='bold')

    ax.set_title(title, color=TEXT, fontsize=9.5, fontweight='bold', pad=6)

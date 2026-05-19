"""
plots_regression.py — Графіки для регресії та візуалізації
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from plots import _style_fig, _style_ax, BG, PANEL, TEXT, GRID, ACCENT, ACCENT2

def regression_diagnostic_figure(res: dict) -> Figure:
    fig = Figure(figsize=(12, 8), dpi=110)
    _style_fig(fig)
    
    # 4 підграфіки (2x2)
    ax1 = fig.add_subplot(2, 2, 1)  # ε vs ŷ (перший вид)
    ax2 = fig.add_subplot(2, 2, 2)  # ε vs ŷ (другий вид)
    ax3 = fig.add_subplot(2, 2, 3)  # ε vs X₁
    ax4 = fig.add_subplot(2, 2, 4)  # ε vs X₂
    
    # Стилізуй кожну вісь
    _style_ax(ax1, "ε vs ŷ (a)", "ŷ", "ε")
    _style_ax(ax2, "ε vs ŷ (б)", "ŷ", "ε")
    _style_ax(ax3, "ε vs X₁ (в)", "X₁", "ε")
    _style_ax(ax4, "ε vs X₂ (г)", "X₂", "ε")
    
    # Дані
    residuals = res['residuals']
    y_hat = res['y_hat']
    X = res.get('X_predictors')  # потрібно додати в stats_regression.py!
    
    # График 1: ε vs ŷ
    ax1.scatter(y_hat, residuals, s=20, color=ACCENT, alpha=0.7)
    ax1.axhline(0, color=ACCENT2, linestyle='--', lw=1)
    ax1.set_facecolor(PANEL)
    
    # График 2: ε vs ŷ (нормовані залишки)
    residuals_std = residuals / np.std(residuals)
    ax2.scatter(y_hat, residuals_std, s=20, color=ACCENT, alpha=0.7)
    ax2.axhline(0, color=ACCENT2, linestyle='--', lw=1)
    ax2.set_facecolor(PANEL)
    
    # График 3: ε vs X₁
    if X is not None and X.shape[1] > 0:
        ax3.scatter(X[:, 0], residuals, s=20, color=ACCENT, alpha=0.7)
        ax3.axhline(0, color=ACCENT2, linestyle='--', lw=1)
    ax3.set_facecolor(PANEL)
    
    # График 4: ε vs X₂
    if X is not None and X.shape[1] > 1:
        ax4.scatter(X[:, 1], residuals, s=20, color=ACCENT, alpha=0.7)
        ax4.axhline(0, color=ACCENT2, linestyle='--', lw=1)
    ax4.set_facecolor(PANEL)
    
    fig.tight_layout()
    return fig

def parallel_coordinates_figure(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(8, 5), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    _style_ax(ax, "Паралельні координати", "", "")

    for i, row in df.iterrows():
        ax.plot(range(df.shape[1]), row, marker='o', alpha=0.6, lw=1)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, color=TEXT)
    fig.tight_layout()
    return fig

def bubble_chart_figure(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 5), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    _style_ax(ax, "Бульбашкова діаграма (X1, X2, розмір = X3)", df.columns[0], df.columns[1])

    sizes = (df.iloc[:, 2] - df.iloc[:, 2].min()) / (df.iloc[:, 2].max() - df.iloc[:, 2].min()) * 300 + 20
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=sizes, alpha=0.7, c=df.iloc[:, 2], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label=df.columns[2])
    fig.tight_layout()
    return fig

def heatmap_data_figure(df: pd.DataFrame) -> Figure:
    """Теплова карта спостережень × ознаки (нормовані значення)"""
    from matplotlib.colors import Normalize
    from stats_regression import normalize_dataframe  # імпортуй з plots або з stats_regression
    
    df_norm = normalize_dataframe(df)
    
    fig = Figure(figsize=(10, 6), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    
    im = ax.imshow(df_norm.T, aspect='auto', cmap='YlGn', interpolation='nearest')
    ax.set_xlabel('Спостереження', color=TEXT)
    ax.set_ylabel('Ознаки', color=TEXT)
    ax.set_title('Теплова карта даних (нормовані значення 0–1)', color=TEXT, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, color=TEXT, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Нормоване значення', color=TEXT)
    
    fig.tight_layout()
    return fig

def glyph_star_figure(df: pd.DataFrame, max_obs: int = 50) -> Figure:
    """
    Зіркові гліфи (Star Glyphs / Star Plots).

    Кожне спостереження зображується як зірка: кожний промінь
    відповідає одній ознаці, довжина променя = нормоване значення [0,1].
    Суміжні кінці промінів з'єднуються — утворюється багатокутник.

    Алгоритм:
      1. Нормуємо кожну ознаку на [0,1]:  v_j = (x_j - min_j)/(max_j - min_j)
      2. Кути промінів рівномірно розподілені: θ_j = 2π·j/p
      3. Координата кінця j-го променя:  (v_j·cos θ_j, v_j·sin θ_j)
      4. Полігон замикається на перший промінь
    """
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Нормування [0, 1]
    df_n = df.copy()
    for col in df_n.columns:
        lo, hi = df_n[col].min(), df_n[col].max()
        df_n[col] = (df_n[col] - lo) / (hi - lo) if hi > lo else 0.0

    n_obs  = min(len(df_n), max_obs)
    p      = len(df_n.columns)
    cols   = list(df_n.columns)

    # Кути промінів
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    # Замикаємо полігон
    angles_c = np.append(angles, angles[0])

    # Розмір сітки
    ncols_g = max(5, int(np.ceil(np.sqrt(n_obs * 1.6))))
    nrows_g = int(np.ceil(n_obs / ncols_g))

    cell   = 1.6
    fig    = Figure(figsize=(ncols_g * cell + 1.2,
                             nrows_g * cell + 1.0), dpi=110)
    _style_fig(fig)
    fig.suptitle(
        f"Зіркові гліфи — {n_obs} спостережень × {p} ознак\n"
        f"Кожний промінь = нормована ознака [0–1]",
        fontsize=10, fontweight='bold', color=TEXT, y=1.01)

    # Колірна палітра для промінів
    spoke_colors = plt.cm.tab20(np.linspace(0, 1, p))

    for idx in range(n_obs):
        row_i = idx // ncols_g
        col_i = idx %  ncols_g

        ax = fig.add_axes([
            0.04 + col_i / ncols_g * (1 - 0.08),
            0.04 + (nrows_g - 1 - row_i) / nrows_g * (1 - 0.10),
            1.0 / ncols_g * 0.88,
            1.0 / nrows_g * 0.82,
        ])
        ax.set_facecolor(PANEL)
        ax.set_aspect('equal')
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis('off')

        values = df_n.iloc[idx].values

        # Кола фону (0.25, 0.5, 0.75, 1.0)
        for r in [0.25, 0.5, 0.75, 1.0]:
            circle = plt.Circle((0, 0), r,
                                 color=GRID, fill=False,
                                 linewidth=0.4, alpha=0.5)
            ax.add_patch(circle)

        # Осьові лінії (промені-сітка)
        for a in angles:
            ax.plot([0, np.cos(a)], [0, np.sin(a)],
                    color=GRID, lw=0.5, alpha=0.6)

        # Полігон гліфа
        xs = np.append(values * np.cos(angles), values[0] * np.cos(angles[0]))
        ys = np.append(values * np.sin(angles), values[0] * np.sin(angles[0]))
        ax.fill(xs, ys, color=ACCENT, alpha=0.30)
        ax.plot(xs, ys, color=ACCENT, lw=1.0, alpha=0.85)

        # Кольорові крапки на кінцях промінів
        for j in range(p):
            ax.scatter(values[j] * np.cos(angles[j]),
                       values[j] * np.sin(angles[j]),
                       s=12, color=spoke_colors[j], zorder=5,
                       edgecolors='none')

        # Номер спостереження
        ax.text(0, -1.18, f"#{idx+1}",
                ha='center', va='top',
                fontsize=6, color=TEXT,
                fontfamily='Consolas')

    # Легенда ознак (один раз, під фігурою)
    legend_ax = fig.add_axes([0.0, 0.0, 1.0, 0.04])
    legend_ax.axis('off')
    legend_ax.set_facecolor(BG)
    x_step = 1.0 / p
    for j, (col_name, clr) in enumerate(zip(cols, spoke_colors)):
        legend_ax.scatter(x_step * j + x_step * 0.15, 0.5,
                          s=30, color=clr,
                          transform=legend_ax.transAxes,
                          clip_on=False)
        legend_ax.text(x_step * j + x_step * 0.22, 0.5,
                       col_name, color=TEXT,
                       fontsize=7, va='center',
                       transform=legend_ax.transAxes,
                       clip_on=False)

    return fig


def spatial_glyph_3d_figure(df: pd.DataFrame) -> Figure:
    """
    3D Гліф: Просторова модель (поверхня) за трьома ознаками.
    Використовує перші дві ознаки як X, Y та третю як Z (інтенсивність).
    """
    if df.shape[1] < 3:
        fig = Figure(figsize=(7, 5), dpi=110)
        _style_fig(fig)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Для 3D-поверхні потрібно\nпринаймні 3 ознаки", 
                ha='center', va='center', color=TEXT, fontfamily='Consolas')
        ax.axis('off')
        return fig

    fig = Figure(figsize=(9, 7), dpi=110)
    _style_fig(fig)
    
    # Створюємо 3D-осі
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    z = df.iloc[:, 2].values

    # Побудова поверхні (Tri-Surface works well for irregular data)
    surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.7, antialiased=True)
    
    # Додаємо точки вимірювання (як на скріншоті)
    ax.scatter(x, y, z, color=ACCENT2, s=15, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Налаштування осей
    ax.set_xlabel(df.columns[0], color=TEXT, fontsize=8, labelpad=5)
    ax.set_ylabel(df.columns[1], color=TEXT, fontsize=8, labelpad=5)
    ax.set_zlabel(df.columns[2], color=TEXT, fontsize=8, labelpad=5)
    ax.set_title(f"3D Гліф: Просторова модель {df.columns[2]}", 
                 color=TEXT, fontsize=11, fontweight='bold', pad=15)

    # Стилізація "панелей" 3D
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color(PANEL)
        axis.label.set_color(TEXT)
        axis.get_ticklines()
    
    ax.tick_params(axis='x', colors=TEXT, labelsize=7)
    ax.tick_params(axis='y', colors=TEXT, labelsize=7)
    ax.tick_params(axis='z', colors=TEXT, labelsize=7)

    # Колірна шкала
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.outline.set_edgecolor(GRID)

    return fig


def regression_prediction_ci_figure(res: dict) -> Figure:
    """Графік: прогнозовані значення з довірчими інтервалами"""
    fig = Figure(figsize=(10, 5), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    
    x = np.arange(len(res['y']))
    ax.scatter(x, res['y'], s=20, color=ACCENT, alpha=0.7, label='Фактичні y')
    ax.plot(x, res['y_hat'], color=ACCENT2, linewidth=1.5, label='Прогнозовані ŷ')
    ax.fill_between(x, res['y_pred_ci_lower'], res['y_pred_ci_upper'], 
                     color=ACCENT, alpha=0.2, label=f"ДІ ({int((1-res['alpha'])*100)}%)")
    
    ax.set_xlabel('Спостереження', color=TEXT)
    ax.set_ylabel('Значення y', color=TEXT)
    ax.set_title(f"Прогнозовані значення для {res['dep_col']} з ДІ", 
                 color=TEXT, fontsize=11, fontweight='bold')
    ax.set_facecolor(PANEL)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    
    fig.tight_layout()
    return fig
"""
tab_lab3.py  — Лабораторна робота №3:
  «Побудова лінійного різноманіття в тривимірному просторі спостережень»

Варіант 6:
  N=300, sigma=11, sigma_shum=2, phi=0.4
  theta1=50 grad (навколо z), theta2=75 grad (навколо y)
  mu=(15, 5, 20)

Порядок виконання:
  1. Генерування / завантаження даних
  2. Первинний аналіз (статистики, гістограми, 3D scatter)
  3. Нелінійне перетворення x3 -> log(x3), повторний аналіз
  4. МГК на кореляційній матриці
  5. Перевірка ГК: некорельованість, дисперсії = власні числа
  6. Оцінка розмірності (m=2), таблиця часток дисперсії
  7. Зворотне перетворення (побудова різноманіття)
  8. Рівняння площини (3 точки -> детермінант)
  9. Візуалізація: вихідні + очищені точки + площина
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from gui_widgets import C, DataTable, MetricCard, SectionHeader, PlotPanel


# ─────────────────────────────────────────────────────────────────
#  ПАРАМЕТРИ ВАРІАНТУ 6
# ─────────────────────────────────────────────────────────────────
VAR6 = {
    'N': 300,
    'sigma': 11.0,
    'sigma_noise': 2.0,
    'phi': 0.4,
    'theta1_deg': 50.0,
    'theta2_deg': 75.0,
    'mu': np.array([15.0, 5.0, 20.0]),
}
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab3_variant6_data.csv')


# ─────────────────────────────────────────────────────────────────
#  МАТЕМАТИЧНЕ ЯДРО
# ─────────────────────────────────────────────────────────────────
def generate_data(params: dict, seed: int = 42) -> pd.DataFrame:
    """
    Генерує вибірку за варіантом.
    1. Xi ~ N(0, sigma^2) для xi1, xi2;  xi3 ~ N(0, sigma_noise^2)
    2. Поворот Ry @ Rz + зсув mu
    3. Асиметрія: x3_obs = exp(phi * (X3 - mean(X3)) / std(X3))
    """
    np.random.seed(seed)
    N = params['N']
    sigma = params['sigma']
    snoise = params['sigma_noise']
    phi = params['phi']
    t1 = np.deg2rad(params['theta1_deg'])
    t2 = np.deg2rad(params['theta2_deg'])
    mu = params['mu']

    xi1 = np.random.normal(0, sigma, N)
    xi2 = np.random.normal(0, sigma, N)
    xi3 = np.random.normal(0, snoise, N)
    Xi = np.column_stack([xi1, xi2, xi3])

    Rz = np.array([[np.cos(t1), -np.sin(t1), 0],
                   [np.sin(t1),  np.cos(t1), 0],
                   [0, 0, 1]])
    Ry = np.array([[ np.cos(t2), 0, np.sin(t2)],
                   [0, 1, 0],
                   [-np.sin(t2), 0, np.cos(t2)]])

    X = (Ry @ Rz @ Xi.T).T + mu

    # Асиметрія третьої ознаки: X3 -> exp(phi*(X3-mean)/std)
    X3_norm = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()
    X[:, 2] = np.exp(phi * X3_norm)

    return pd.DataFrame(X, columns=['x1', 'x2', 'x3'])


def primary_stats(df: pd.DataFrame) -> list:
    """Таблиця первинних статистик: середнє, дисперсія, асиметрія, ексцес."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            'Ознака': col,
            'Середнє': f'{s.mean():.4f}',
            'Дисперсія': f'{s.var(ddof=1):.4f}',
            'СКВ': f'{s.std(ddof=1):.4f}',
            'Асиметрія': f'{stats.skew(s):.4f}',
            'Ексцес': f'{stats.kurtosis(s):.4f}',
        })
    return rows


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Логарифмує ознаки, у яких |skewness| > 0.5."""
    df2 = df.copy()
    transformed = {}
    for col in df.columns:
        sk = stats.skew(df[col])
        if abs(sk) > 0.5 and df[col].min() > 0:
            df2[col] = np.log(df[col])
            transformed[col] = sk
    return df2, transformed


def run_pca(df: pd.DataFrame):
    """
    МГК на кореляційній матриці.
    Повертає словник з усіма результатами.
    """
    X = df.values.astype(float)
    n, p = X.shape
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)
    X_std[X_std == 0] = 1.0
    X_z = (X - X_mean) / X_std

    R = np.corrcoef(X_z.T)

    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Нормалізація знаку (найбільший за модулем елемент – додатній)
    for j in range(p):
        mi = np.argmax(np.abs(evecs[:, j]))
        if evecs[mi, j] < 0:
            evecs[:, j] *= -1.0

    var_pct = evals / evals.sum() * 100.0
    cum_pct = np.cumsum(var_pct)

    Z = X_z @ evecs   # (n x p) – координати в системі ГК

    # Таблиця власних векторів + λ + %
    pc_names = [f'PC{i+1}' for i in range(p)]
    tbl_rows = []
    for i, feat in enumerate(df.columns):
        row = {'Ознака': feat}
        for j in range(p):
            row[pc_names[j]] = f'{evecs[i,j]:+.4f}'
        tbl_rows.append(row)
    tbl_rows.append({'Ознака': 'λ (власне число)', **{pc_names[j]: f'{evals[j]:.4f}' for j in range(p)}})
    tbl_rows.append({'Ознака': 'Частка дисперсії, %', **{pc_names[j]: f'{var_pct[j]:.2f}%' for j in range(p)}})
    tbl_rows.append({'Ознака': 'Накопичена, %', **{pc_names[j]: f'{cum_pct[j]:.2f}%' for j in range(p)}})

    return {
        'n': n, 'p': p, 'cols': list(df.columns),
        'X': X, 'X_mean': X_mean, 'X_std': X_std, 'X_z': X_z,
        'R': R, 'evals': evals, 'evecs': evecs,
        'var_pct': var_pct, 'cum_pct': cum_pct,
        'Z': Z, 'pc_names': pc_names, 'tbl_rows': tbl_rows,
    }


def build_manifold(res: dict, m: int = 2):
    """
    Зворотне перетворення за m першими ГК -> очищені точки.
    Повертає очищені точки і рівняння площини (якщо m=2).
    """
    Z_m = res['Z'][:, :m]
    V_m = res['evecs'][:, :m]

    # Відновлені (очищені) точки у стандартизованому просторі
    X_clean_z = Z_m @ V_m.T                             # (n x p)
    # Повернення у вихідний масштаб
    X_clean = X_clean_z * res['X_std'] + res['X_mean']

    # Нормаль до площини = (p+1)-й власний вектор (що кидається за m-ою ГК)
    # У стандартизованому просторі
    n_std = res['evecs'][:, m]   # вектор нормалі (стандартизовані координати)

    # Коефіцієнти рівняння площини у нестандартизованому просторі:
    # sum_j (n_std[j] / X_std[j]) * (X_j - X_mean[j]) = 0
    A_coef = n_std / res['X_std']
    D_coef = -np.dot(A_coef, res['X_mean'])

    # Перевірка через детермінант (3 точки з очищеного масиву)
    # Вибираємо 3 точки з максимально рознесеними позиціями
    idx_sel = _pick_noncollinear(X_clean_z, m)
    p1 = X_clean_z[idx_sel[0]]
    p2 = X_clean_z[idx_sel[1]]
    p3 = X_clean_z[idx_sel[2]]
    n_det = np.cross(p2 - p1, p3 - p1)
    D_det_std = -np.dot(n_det, p1)

    # Максимальне відхилення очищених точок від площини
    residuals = X_clean_z @ n_std
    rmse_manifold = float(np.sqrt(np.mean(residuals**2)))

    return {
        'm': m,
        'X_clean_z': X_clean_z,
        'X_clean': X_clean,
        'n_std': n_std,
        'A_coef': A_coef,
        'D_coef': D_coef,
        'rmse_manifold': rmse_manifold,
        'det_p1': p1, 'det_p2': p2, 'det_p3': p3,
        'n_det': n_det, 'D_det_std': D_det_std,
    }


def _pick_noncollinear(X_clean_z: np.ndarray, m: int, n_try: int = 200) -> list:
    """Вибирає 3 точки з найбільшою площею трикутника (не колінеарні)."""
    n = len(X_clean_z)
    best_area = -1.0
    best = [0, 1, 2]
    rng = np.random.default_rng(0)
    for _ in range(n_try):
        idx = rng.choice(n, 3, replace=False).tolist()
        p1, p2, p3 = X_clean_z[idx[0]], X_clean_z[idx[1]], X_clean_z[idx[2]]
        area = np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        if area > best_area:
            best_area = area
            best = idx
    return best


# ─────────────────────────────────────────────────────────────────
#  ГРАФІКИ
# ─────────────────────────────────────────────────────────────────
BG = '#0f1117'
PANEL = '#1a1d27'
TEXT = '#d6dae8'
GRID = '#252836'
ACCENT = '#4f8ef7'
ACCENT2 = '#e05c7a'
GREEN = '#4ecb88'
YELLOW = '#f5c542'


def _sax(ax, title='', xl='', yl=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.6)
    if title: ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    if xl:    ax.set_xlabel(xl, fontsize=8)
    if yl:    ax.set_ylabel(yl, fontsize=8)


def histograms_figure(df: pd.DataFrame, title: str = 'Гістограми ознак') -> Figure:
    p = len(df.columns)
    fig = Figure(figsize=(5 * p, 4), dpi=100)
    fig.patch.set_facecolor(BG)
    for j, col in enumerate(df.columns):
        ax = fig.add_subplot(1, p, j + 1)
        _sax(ax, f'{col}  (skew={stats.skew(df[col]):.3f})', col, 'Частота')
        bins = max(10, int(np.sqrt(len(df))))
        ax.hist(df[col].values, bins=bins, color=ACCENT, edgecolor=BG, alpha=0.85)
        # Крива нормального розподілу
        mu_j = df[col].mean(); sg_j = df[col].std()
        xs = np.linspace(df[col].min(), df[col].max(), 200)
        ys = stats.norm.pdf(xs, mu_j, sg_j) * len(df) * (df[col].max() - df[col].min()) / bins
        ax.plot(xs, ys, color=ACCENT2, lw=1.8, label='N(μ,σ)')
        ax.axvline(mu_j, color=YELLOW, lw=1.2, ls='--', alpha=0.8, label=f'μ={mu_j:.2f}')
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    fig.suptitle(title, color=TEXT, fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def scatter3d_figure(X_raw: np.ndarray, cols: list,
                     title: str = '3D-діаграма розсіювання',
                     color=ACCENT, alpha=0.65) -> Figure:
    fig = Figure(figsize=(9, 7), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection='3d')
    _style_3d_ax(ax)
    ax.scatter(X_raw[:, 0], X_raw[:, 1], X_raw[:, 2],
               c=X_raw[:, 0], cmap='plasma', s=22, alpha=alpha, edgecolors='none')
    ax.set_xlabel(cols[0], color=TEXT, fontsize=9, labelpad=7)
    ax.set_ylabel(cols[1], color=TEXT, fontsize=9, labelpad=7)
    ax.set_zlabel(cols[2], color=TEXT, fontsize=9, labelpad=7)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=12)
    ax.view_init(elev=24, azim=46)
    fig.tight_layout()
    return fig


def scree_figure(evals: np.ndarray, var_pct: np.ndarray, cum_pct: np.ndarray,
                 m_chosen: int = 2) -> Figure:
    p = len(evals)
    x = np.arange(1, p + 1)
    fig = Figure(figsize=(10, 4.5), dpi=100)
    fig.patch.set_facecolor(BG)

    ax1 = fig.add_subplot(1, 2, 1)
    _sax(ax1, 'Scree plot (власні числа λ)', 'ГК', 'λ')
    colors = [GREEN if i < m_chosen else ACCENT for i in range(p)]
    ax1.bar(x, evals, color=colors, alpha=0.8, edgecolor=BG)
    ax1.plot(x, evals, 'o-', color=YELLOW, lw=1.8, markersize=6)
    ax1.axhline(1.0, color=ACCENT2, ls='--', lw=1.2, label='λ=1 (Кайзер)')
    ax1.set_xticks(x); ax1.set_xticklabels([f'PC{i}' for i in x], fontsize=8)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    for xi, lam in zip(x, evals):
        ax1.text(xi, lam + max(evals) * 0.03, f'{lam:.3f}', ha='center', fontsize=8, color=YELLOW, fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2)
    _sax(ax2, 'Пояснена дисперсія та накопичена частка', 'ГК', '%')
    ax2.bar(x, var_pct, color=ACCENT, alpha=0.7, edgecolor=BG, label='% на ГК')
    ax2.plot(x, cum_pct, 's-', color=GREEN, lw=2, markersize=6, label='Накопичена %')
    ax2.axhline(95, color=ACCENT2, ls='--', lw=1.2, label='95% поріг')
    ax2.axvline(m_chosen + 0.45, color=YELLOW, ls=':', lw=1.3, label=f'Вибрано m={m_chosen}')
    ax2.set_ylim(0, 115); ax2.set_xticks(x)
    ax2.set_xticklabels([f'PC{i}' for i in x], fontsize=8)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc='center right')
    for xi, pct, cum in zip(x, var_pct, cum_pct):
        ax2.text(xi, pct + 2, f'{pct:.1f}%', ha='center', fontsize=8, color=TEXT)
        ax2.text(xi, cum + 2, f'{cum:.1f}%', ha='center', fontsize=8, color=GREEN, fontweight='bold')

    fig.tight_layout()
    return fig


def corr_heatmap_figure(R: np.ndarray, labels: list) -> Figure:
    from matplotlib.colors import LinearSegmentedColormap
    p = len(labels)
    cmap = LinearSegmentedColormap.from_list('corr',
        [(0.87, 0.22, 0.38), (0.12, 0.13, 0.20), (0.31, 0.56, 0.97)], N=256)
    fig = Figure(figsize=(max(5, p * 1.5), max(4, p * 1.3)), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    ax.set_xticks(range(p)); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9, color=TEXT)
    ax.set_yticks(range(p)); ax.set_yticklabels(labels, fontsize=9, color=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    for i in range(p):
        for j in range(p):
            v = R[i, j]
            fg = 'white' if abs(v) > 0.45 else TEXT
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9, color=fg, fontweight='bold')
    ax.set_title('Кореляційна матриця R', color=TEXT, fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


def manifold_figure(res: dict, man: dict, log_transformed: bool = True,
                    df_original: pd.DataFrame = None) -> Figure:
    """
    3D-візуалізація:
    - Вихідні точки (сині)
    - Очищені точки (зелені)
    - Площина різноманіття (напівпрозора)
    """
    fig = Figure(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection='3d')
    _style_3d_ax(ax)

    X_orig = res['X_z']             # стандартизований простір (після log)
    X_clean = man['X_clean_z']      # очищені (стандартизований)
    cols = res['cols']

    ax.scatter(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2],
               c=ACCENT, s=15, alpha=0.45, edgecolors='none',
               label=f'Вихідні точки (n={res["n"]})', depthshade=True)

    ax.scatter(X_clean[:, 0], X_clean[:, 1], X_clean[:, 2],
               c=GREEN, s=20, alpha=0.80, edgecolors='none',
               label='Очищені точки (різноманіття)', depthshade=True)

    # Площина: z_std = -(A/C) * x_std - (B/C) * y_std
    n = man['n_std']
    if abs(n[2]) > 1e-10:
        x_range = np.array([X_orig[:, 0].min(), X_orig[:, 0].max()])
        y_range = np.array([X_orig[:, 1].min(), X_orig[:, 1].max()])
        xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 20),
                             np.linspace(y_range[0], y_range[1], 20))
        zz = (-n[0] * xx - n[1] * yy) / n[2]
        ax.plot_surface(xx, yy, zz, alpha=0.22, color=ACCENT2, label='Площина різноманіття')

    ax.set_xlabel(f'{cols[0]}(std)', color=TEXT, fontsize=9, labelpad=7)
    ax.set_ylabel(f'{cols[1]}(std)', color=TEXT, fontsize=9, labelpad=7)
    ax.set_zlabel(f'{cols[2]}(std)', color=TEXT, fontsize=9, labelpad=7)

    suffix = ' [log-просторі]' if log_transformed else ''
    ax.set_title(f'Лінійне різноманіття (площина) у стандартизованому{suffix} просторі\n'
                 f'm=2 ГК | RMSE={man["rmse_manifold"]:.2e}',
                 color=TEXT, fontsize=10, fontweight='bold', pad=12)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc='upper right')
    ax.view_init(elev=24, azim=46)

    fig._lab3_ax = ax
    fig.tight_layout()
    return fig


def pc_check_figure(res: dict) -> Figure:
    """Графіки перевірки: scatter PC1 vs PC2, PC1 vs PC3, дисперсії."""
    Z = res['Z']
    p = res['p']
    pc_names = res['pc_names']
    fig = Figure(figsize=(12, 4), dpi=100)
    fig.patch.set_facecolor(BG)

    pairs = [(0, 1), (0, 2)] if p >= 3 else [(0, 1)]
    ncols = len(pairs) + 1
    for idx, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(1, ncols, idx + 1)
        _sax(ax, f'{pc_names[i]} vs {pc_names[j]}', pc_names[i], pc_names[j])
        ax.scatter(Z[:, i], Z[:, j], s=12, color=ACCENT, alpha=0.55, edgecolors='none')
        r_val = np.corrcoef(Z[:, i], Z[:, j])[0, 1]
        ax.set_title(f'{pc_names[i]} vs {pc_names[j]}  (r={r_val:.4f})', color=TEXT, fontsize=9, fontweight='bold')

    ax_v = fig.add_subplot(1, ncols, ncols)
    _sax(ax_v, 'Дисперсії ГК vs λ', 'ГК', 'Дисперсія')
    x = np.arange(1, p + 1)
    var_z = np.var(Z, axis=0, ddof=1)
    ax_v.bar(x - 0.2, res['evals'], 0.35, color=ACCENT, alpha=0.8, label='λ (власне число)')
    ax_v.bar(x + 0.2, var_z, 0.35, color=GREEN, alpha=0.8, label='var(PCk)')
    ax_v.set_xticks(x); ax_v.set_xticklabels(pc_names, fontsize=8)
    ax_v.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    fig.suptitle('Перевірка ГК: некорельованість та var(PCk) = λk', color=TEXT, fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


def _style_3d_ax(ax):
    ax.set_facecolor(BG)
    pane_c = (0.10, 0.11, 0.16, 1.0)
    edge_c = (0.16, 0.18, 0.25, 1.0)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor(pane_c)
        pane.set_edgecolor(edge_c)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.grid(True, color=GRID, linestyle=':', alpha=0.5)


# ─────────────────────────────────────────────────────────────────
#  GUI ВКЛАДКА
# ─────────────────────────────────────────────────────────────────
class TabLab3(ttk.Frame):
    """Вкладка Лаб-3: побудова лінійного різноманіття."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.df_raw = None      # завантажені / згенеровані дані (до log)
        self.df_log = None      # після log-transform
        self.pca_res = None
        self.manifold = None
        self._fig_manifold = None
        self._build()
        if os.path.exists(DEFAULT_CSV):
            self.after(150, self._run)

    # ──────────────────────────────────────────────────────────
    def _build(self):
        # ── Верхня панель ─────────────────────────────────────
        top = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        top.pack(fill='x')

        title_box = tk.Frame(top, bg=C['bg'])
        title_box.pack(side='left')
        tk.Label(title_box,
                 text="Лаб-3: Лінійне різноманіття в 3D (Варіант 6)",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 13, 'bold')).pack(anchor='w')
        tk.Label(title_box,
                 text="N=300  sigma=11  sigma_shum=2  phi=0.4  theta1=50 deg  theta2=75 deg  mu=(15,5,20)",
                 bg=C['bg'], fg=C['subtext'],
                 font=('Consolas', 8)).pack(anchor='w')

        # Кнопки
        btn_box = tk.Frame(top, bg=C['bg'])
        btn_box.pack(side='right')
        ttk.Button(btn_box, text="Генерувати дані",
                   command=self._generate).pack(side='left', padx=4)
        ttk.Button(btn_box, text="Завантажити CSV…",
                   command=self._browse).pack(side='left', padx=4)
        self.path_var = tk.StringVar(value=DEFAULT_CSV)
        tk.Entry(btn_box, textvariable=self.path_var, width=22,
                 bg=C['panel2'], fg=C['text'],
                 insertbackground=C['accent'],
                 relief='flat').pack(side='left', padx=4)
        ttk.Button(btn_box, text="▶  Розрахувати",
                   style='Accent.TButton',
                   command=self._run).pack(side='left', padx=4)

        ttk.Separator(self).pack(fill='x')

        # ── Картки підсумків ───────────────────────────────────
        self.cards = tk.Frame(self, bg=C['bg'])
        self.cards.pack(fill='x', padx=12, pady=(6, 2))

        # ── Основний notebook ──────────────────────────────────
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=8, pady=8)

        self._build_tabs()

    def _build_tabs(self):
        self.t_data    = ttk.Frame(self.nb)
        self.t_hist    = ttk.Frame(self.nb)
        self.t_hist_log = ttk.Frame(self.nb)
        self.t_pca     = ttk.Frame(self.nb)
        self.t_check   = ttk.Frame(self.nb)
        self.t_scree   = ttk.Frame(self.nb)
        self.t_plane   = ttk.Frame(self.nb)
        self.t_3d      = ttk.Frame(self.nb)

        self.nb.add(self.t_data,     text="  1. Дані та статистика  ")
        self.nb.add(self.t_hist,     text="  2. Гістограми (до)  ")
        self.nb.add(self.t_hist_log, text="  3. Гістограми (після log)  ")
        self.nb.add(self.t_pca,      text="  4. МГК (R, λ, V)  ")
        self.nb.add(self.t_check,    text="  5. Перевірка ГК  ")
        self.nb.add(self.t_scree,    text="  6. Scree plot  ")
        self.nb.add(self.t_plane,    text="  7. Рівняння площини  ")
        self.nb.add(self.t_3d,       text="  8. Візуалізація 3D  ")

        # Вкладка 1: Дані та статистика
        SectionHeader(self.t_data, "  Первинна статистика ознак").pack(fill='x', padx=4, pady=(4, 0))
        self._tbl_stats = ttk.Frame(self.t_data)
        self._tbl_stats.pack(fill='both', expand=True, padx=4, pady=4)

        # Вкладки графіків
        self.pp_hist     = PlotPanel(self.t_hist)
        self.pp_hist.pack(fill='both', expand=True)

        self.pp_hist_log = PlotPanel(self.t_hist_log)
        self.pp_hist_log.pack(fill='both', expand=True)

        # Вкладка 4: МГК
        pca_paned = ttk.PanedWindow(self.t_pca, orient='vertical')
        pca_paned.pack(fill='both', expand=True, padx=4, pady=4)
        self._pca_top = ttk.Frame(pca_paned)   # кор. матриця
        self._pca_bot = ttk.Frame(pca_paned)   # таблиця
        pca_paned.add(self._pca_top, weight=2)
        pca_paned.add(self._pca_bot, weight=3)
        self.pp_corr = PlotPanel(self._pca_top)
        self.pp_corr.pack(fill='both', expand=True)
        SectionHeader(self._pca_bot, "  Таблиця власних векторів, λ та часток дисперсії").pack(fill='x', padx=4, pady=(4,0))
        self._tbl_pca = ttk.Frame(self._pca_bot)
        self._tbl_pca.pack(fill='both', expand=True, padx=4, pady=4)

        # Вкладка 5: Перевірка ГК
        self.pp_check = PlotPanel(self.t_check)
        self.pp_check.pack(fill='both', expand=True)

        # Вкладка 6: Scree
        self.pp_scree = PlotPanel(self.t_scree)
        self.pp_scree.pack(fill='both', expand=True)

        # Вкладка 7: Рівняння площини
        self._build_plane_tab()

        # Вкладка 8: 3D
        ctrl_3d = tk.Frame(self.t_3d, bg=C['bg'], pady=4, padx=8)
        ctrl_3d.pack(fill='x')
        tk.Label(ctrl_3d, text="Кут огляду:", bg=C['bg'], fg=C['text'],
                 font=('Consolas', 9)).pack(side='left', padx=(0,6))
        for label, (el, az) in [("Ізометрія", (24, 46)),
                                 ("Зверху (XY)", (90, -90)),
                                 ("Збоку (XZ)", (0, -90)),
                                 ("Фронтально (YZ)", (0, 0))]:
            ttk.Button(ctrl_3d, text=label,
                       command=lambda e=el, a=az: self._set_3d_view(e, a)).pack(side='left', padx=2)
        self.pp_3d = PlotPanel(self.t_3d)
        self.pp_3d.pack(fill='both', expand=True)

    def _build_plane_tab(self):
        frame = tk.Frame(self.t_plane, bg=C['bg'], padx=20, pady=16)
        frame.pack(fill='both', expand=True)

        SectionHeader(frame, "  Рівняння площини різноманіття", accent=True).pack(fill='x', pady=(0, 10))

        self.plane_text = tk.Text(frame, bg=C['panel2'], fg=C['text'],
                                  font=('Consolas', 10), relief='flat',
                                  height=20, insertbackground=C['accent'],
                                  state='disabled')
        self.plane_text.pack(fill='both', expand=True)

    # ──────────────────────────────────────────────────────────
    def _generate(self):
        """Генерує дані за варіантом 6 та зберігає CSV."""
        df = generate_data(VAR6, seed=42)
        path = self.path_var.get().strip() or DEFAULT_CSV
        df.to_csv(path, index=False, float_format='%.6f')
        self.df_raw = df
        self.app.status.ok(f"Дані згенеровано: {len(df)} спостережень. Збережено у {path}")
        messagebox.showinfo("Генерація", f"Дані успішно згенеровано та збережено:\n{path}")

    def _browse(self):
        f = filedialog.askopenfilename(
            title="Відкрити файл даних (CSV)",
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All", "*.*")])
        if f:
            self.path_var.set(f)
            self.df_raw = None

    def _load_data(self):
        path = self.path_var.get().strip()
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            if df.shape[1] != 3:
                raise ValueError(f"Файл повинен мати рівно 3 стовпці, отримано: {df.shape[1]}")
            df.columns = ['x1', 'x2', 'x3']
            return df
        if self.df_raw is not None:
            return self.df_raw
        raise ValueError(f"Файл не знайдено: {path}. Натисніть «Генерувати дані» або вкажіть коректний шлях.")

    def _run(self):
        try:
            df_raw = self._load_data()
        except Exception as exc:
            messagebox.showerror("Помилка завантаження", str(exc))
            self.app.status.err(str(exc))
            return

        self.df_raw = df_raw

        # Лог-перетворення
        df_log, transformed = apply_log_transform(df_raw)
        self.df_log = df_log

        # МГК
        pca_res = run_pca(df_log)
        self.pca_res = pca_res

        # m = 2 (задано умовою задачі – площина)
        m = 2
        manifold = build_manifold(pca_res, m=m)
        self.manifold = manifold

        self._fill_ui(df_raw, df_log, transformed, pca_res, manifold)
        self.app.status.ok(
            f"Лаб-3: МГК виконано. λ={[f'{v:.3f}' for v in pca_res['evals']]}  "
            f"RMSE площини={manifold['rmse_manifold']:.2e}")

    def _fill_ui(self, df_raw, df_log, transformed, pca_res, manifold):
        # ── Картки ────────────────────────────────────────────
        for w in self.cards.winfo_children():
            w.destroy()
        MetricCard(self.cards, "Спостережень", pca_res['n']).pack(side='left', padx=4)
        MetricCard(self.cards, "Ознак", pca_res['p']).pack(side='left', padx=4)
        MetricCard(self.cards, "Log-перетворено",
                   ", ".join(transformed.keys()) if transformed else "—",
                   C['yellow']).pack(side='left', padx=4)
        MetricCard(self.cards, "m (розмірність різноманіття)", f"{manifold['m']}",
                   C['green']).pack(side='left', padx=4)
        MetricCard(self.cards, "PC1+PC2 пояснюють",
                   f"{pca_res['cum_pct'][1]:.1f}%", C['green']).pack(side='left', padx=4)
        MetricCard(self.cards, "RMSE площини (std)",
                   f"{manifold['rmse_manifold']:.2e}").pack(side='left', padx=4)

        # ── Таблиця статистики ─────────────────────────────────
        for w in self._tbl_stats.winfo_children():
            w.destroy()
        SectionHeader(self._tbl_stats, "  Вихідні дані (до перетворення)").pack(fill='x')
        DataTable(self._tbl_stats,
                  columns=['Ознака', 'Середнє', 'Дисперсія', 'СКВ', 'Асиметрія', 'Ексцес'],
                  data=primary_stats(df_raw)).pack(fill='x', padx=4, pady=(2, 8))

        SectionHeader(self._tbl_stats, "  Після нелінійного перетворення").pack(fill='x')
        DataTable(self._tbl_stats,
                  columns=['Ознака', 'Середнє', 'Дисперсія', 'СКВ', 'Асиметрія', 'Ексцес'],
                  data=primary_stats(df_log)).pack(fill='x', padx=4, pady=(2, 4))

        # ── Гістограми ─────────────────────────────────────────
        self.pp_hist.show(histograms_figure(df_raw, 'Гістограми вихідних ознак (до перетворення)'))
        self.pp_hist_log.show(histograms_figure(df_log, 'Гістограми після log(x3)'))

        # ── МГК: кореляційна матриця + таблиця ──────────────────
        self.pp_corr.show(corr_heatmap_figure(pca_res['R'], pca_res['cols']))
        for w in self._tbl_pca.winfo_children():
            w.destroy()
        DataTable(self._tbl_pca,
                  columns=['Ознака'] + pca_res['pc_names'],
                  data=pca_res['tbl_rows']).pack(fill='both', expand=True)

        # ── Перевірка ГК ───────────────────────────────────────
        self.pp_check.show(pc_check_figure(pca_res))

        # ── Scree plot ─────────────────────────────────────────
        self.pp_scree.show(scree_figure(pca_res['evals'], pca_res['var_pct'],
                                        pca_res['cum_pct'], m_chosen=manifold['m']))

        # ── Рівняння площини ────────────────────────────────────
        self._fill_plane_text(pca_res, manifold, transformed)

        # ── 3D візуалізація ────────────────────────────────────
        fig_3d = manifold_figure(pca_res, manifold, log_transformed=bool(transformed))
        self._fig_manifold = fig_3d
        self.pp_3d.show(fig_3d)

        self.nb.select(0)

    def _fill_plane_text(self, pca_res, manifold, transformed):
        cols = pca_res['cols']
        n_std = manifold['n_std']
        A = manifold['A_coef']
        D = manifold['D_coef']
        X_mean = pca_res['X_mean']
        X_std = pca_res['X_std']
        m = manifold['m']

        lines = []
        lines.append("=" * 62)
        lines.append("  РІВНЯННЯ ПЛОЩИНИ ЛІНІЙНОГО РІЗНОМАНІТТЯ (m = 2)")
        lines.append("=" * 62)
        lines.append("")
        lines.append("Алгоритм: нормаль до площини = власний вектор v_{m+1}")
        lines.append(f"(вектор, що відповідає найменшому власному числу λ_{m+1})")
        lines.append("")

        # Нормаль у стандартизованому просторі
        lines.append(f"Нормальний вектор у стандартизованому просторі:")
        for j, col in enumerate(cols):
            lines.append(f"  n_std[{col}] = {n_std[j]:+.5f}")
        lines.append("")

        # Рівняння у стандартизованому просторі: n_std dot x_std = 0
        parts_std = []
        for j, col in enumerate(cols):
            parts_std.append(f"({n_std[j]:+.4f})*{col}_std")
        eq_std = " + ".join(parts_std)
        lines.append("Рівняння у стандартизованому просторі:")
        lines.append(f"  {eq_std}  =  0")
        lines.append("")

        # Рівняння у логарифмованому просторі (якщо є log-перетворення)
        log_cols = list(transformed.keys()) if transformed else []
        col_labels = []
        for col in cols:
            if col in log_cols:
                col_labels.append(f"log({col})")
            else:
                col_labels.append(col)

        parts_log = []
        for j, lbl in enumerate(col_labels):
            parts_log.append(f"({A[j]:+.5f})*{lbl}")
        eq_log = " + ".join(parts_log)
        lines.append(f"Рівняння у {'log-' if log_cols else ''}просторі:")
        lines.append(f"  {eq_log}  +  ({D:+.5f})  =  0")
        lines.append("")
        lines.append(f"  де D = {D:+.6f}")
        lines.append("")

        # Три точки, що задають площину (via детермінант)
        lines.append("=" * 62)
        lines.append("  ПЕРЕВІРКА: 3 точки, що задають площину")
        lines.append("  (умова: det рівний нулю)")
        lines.append("=" * 62)
        lines.append("")

        idx_pts = [manifold['det_p1'], manifold['det_p2'], manifold['det_p3']]
        for k, pt in enumerate(idx_pts):
            pt_str = ", ".join(f"{v:+.4f}" for v in pt)
            lines.append(f"  P{k+1} = ({pt_str})  [std]")
        lines.append("")

        n_det = manifold['n_det']
        norm_det = np.linalg.norm(n_det)
        if norm_det > 1e-10:
            n_det_norm = n_det / norm_det
        else:
            n_det_norm = n_det
        lines.append("Нормаль (via cross product двох ребер трикутника P1P2P3):")
        lines.append(f"  n_det = {np.round(n_det, 5).tolist()}")
        lines.append(f"  |n_det| = {norm_det:.6f}")
        lines.append(f"  n_det (нормалізований) = {np.round(n_det_norm, 5).tolist()}")
        lines.append("")

        # Узгоджуємо знаки
        dot_check = np.dot(n_std, n_det_norm)
        lines.append(f"  Cos кута між n_std та n_det: {abs(dot_check):.6f}")
        lines.append(f"  (якщо ~1.0 — вектори паралельні, площини збігаються)")
        lines.append("")

        lines.append("=" * 62)
        lines.append("  ПІДСУМОК: RMSE відхилення очищених точок від площини")
        lines.append(f"  RMSE = {manifold['rmse_manifold']:.6e}  (практично 0, ТАК)")
        lines.append("=" * 62)

        # Параметри варіанта
        lines.append("")
        lines.append("=" * 62)
        lines.append("  ПАРАМЕТРИ ВАРІАНТУ 6")
        lines.append("=" * 62)
        for j, col in enumerate(cols):
            lines.append(f"  {col}: mean = {X_mean[j]:+.4f},  std = {X_std[j]:.4f}")
        lines.append("")
        lines.append(f"  λ1 = {pca_res['evals'][0]:.4f}  ({pca_res['var_pct'][0]:.2f}%)")
        lines.append(f"  λ2 = {pca_res['evals'][1]:.4f}  ({pca_res['var_pct'][1]:.2f}%)")
        lines.append(f"  λ3 = {pca_res['evals'][2]:.4f}  ({pca_res['var_pct'][2]:.2f}%)")
        lines.append(f"  Перші m=2 ГК: {pca_res['cum_pct'][1]:.2f}% дисперсії")
        lines.append(f"  Дисперсія 3-ї ГК (шум): {pca_res['evals'][2]:.4f}")

        txt = "\n".join(lines)
        self.plane_text.config(state='normal')
        self.plane_text.delete('1.0', 'end')
        self.plane_text.insert('end', txt)
        self.plane_text.config(state='disabled')

    def _set_3d_view(self, elev, azim):
        if self._fig_manifold is not None and hasattr(self._fig_manifold, '_lab3_ax'):
            self._fig_manifold._lab3_ax.view_init(elev=elev, azim=azim)
            if self.pp_3d._canvas:
                self.pp_3d._canvas.draw_idle()

"""
stats_pca.py — Метод головних компонент (МГК) для Лаб. 5, пункт 5.

Алгоритм (за посібником, розд. 5):
  1. Стандартизація даних (μ=0, σ=1)
  2. Кореляційна матриця R стандартизованих даних
  3. Власні числа λ та власні вектори V матриці R
  4. Сортування λ за спаданням
  5. Координати в системі ГК: Z = X_std @ V
  6. Зворотній перехід: X_rec = Z @ V.T * σ + μ
"""

import numpy as np
import pandas as pd
from scipy import stats


def pca_analysis(df: pd.DataFrame, selected_cols: list,
                 alpha: float = 0.05) -> dict:
    """
    Повне МГК-перетворення.

    Параметри
    ----------
    df            : вхідний DataFrame
    selected_cols : ознаки, які включаються в аналіз
    alpha         : рівень значущості

    Повертає словник з усіма проміжними та кінцевими результатами.
    """
    X = df[selected_cols].values.astype(float)
    n, p = X.shape

    # ── 1. Центрування та стандартизація ──────────────────────
    X_mean     = X.mean(axis=0)
    X_std_vals = X.std(axis=0, ddof=1)
    X_std_vals = np.where(X_std_vals == 0, 1.0, X_std_vals)   # захист від 0
    X_standardized = (X - X_mean) / X_std_vals

    # ── 2. Кореляційна матриця (на стандартизованих) ──────────
    R_orig = np.corrcoef(X_standardized.T)

    # ── 3. Власні числа та власні вектори ─────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(R_orig)

    # Сортуємо за спаданням λ
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # ── 4. Частки дисперсії ───────────────────────────────────
    total_var = eigenvalues.sum()
    var_pct   = eigenvalues / total_var * 100
    cum_pct   = np.cumsum(var_pct)

    # ── 5. Координати в системі МГК ───────────────────────────
    Z = X_standardized @ eigenvectors       # (n × p)

    # ── 6. Зворотній перехід ──────────────────────────────────
    X_rec_std = Z @ eigenvectors.T          # стандартизований простір
    X_rec     = X_rec_std * X_std_vals + X_mean   # оригінальний масштаб

    # ── Кореляційні матриці ───────────────────────────────────
    R_pca = np.corrcoef(Z.T) if p > 1 else np.array([[1.0]])
    R_rec = np.corrcoef(X_rec.T) if p > 1 else np.array([[1.0]])

    # ── Центри розсіювання ────────────────────────────────────
    center_orig = X_mean.copy()
    center_pca  = Z.mean(axis=0)            # ≈ 0
    center_rec  = X_rec.mean(axis=0)        # ≈ X_mean

    # ── Таблиця МГК (формат Таблиці 4.5 посібника) ───────────
    pc_names  = [f"x'{i+1}" for i in range(p)]
    table_rows = []

    # Рядки навантажень (факторні навантаження = компоненти власних векторів)
    for i, col in enumerate(selected_cols):
        row = {'Ознака': col}
        for j in range(p):
            row[pc_names[j]] = round(float(eigenvectors[i, j]), 3)
        table_rows.append(row)

    # Додаткові рядки: власні числа, % та накопичений %
    ev_row  = {'Ознака': 'Власні числа'}
    pct_row = {'Ознака': '% на напрям'}
    cum_row = {'Ознака': 'Накопичений %'}
    for j in range(p):
        ev_row[pc_names[j]]  = round(float(eigenvalues[j]), 3)
        pct_row[pc_names[j]] = round(float(var_pct[j]),     1)
        cum_row[pc_names[j]] = round(float(cum_pct[j]),     1)
    table_rows += [ev_row, pct_row, cum_row]

    return {
        # ── Таблиця ──────────────────────────────────────────
        'table_rows':   table_rows,
        'pc_names':     pc_names,
        'cols':         selected_cols,

        # ── Розмірності ──────────────────────────────────────
        'n': n,
        'p': p,

        # ── Власні числа та вектори ───────────────────────────
        'eigenvalues':  eigenvalues,
        'eigenvectors': eigenvectors,
        'var_pct':      var_pct,
        'cum_pct':      cum_pct,

        # ── Дані: оригінал / МГК / відновлені ────────────────
        'X_orig':          X,
        'X_standardized':  X_standardized,
        'Z':               Z,
        'X_reconstructed': X_rec,

        # ── Кореляційні матриці ───────────────────────────────
        'R_orig': R_orig,
        'R_pca':  R_pca,
        'R_rec':  R_rec,

        # ── Центри розсіювання ────────────────────────────────
        'center_orig': center_orig,
        'center_pca':  center_pca,
        'center_rec':  center_rec,

        # ── Параметри нормування (для зворотного перетворення) ─
        'X_mean':      X_mean,
        'X_std_vals':  X_std_vals,
        'alpha':       alpha,
    }
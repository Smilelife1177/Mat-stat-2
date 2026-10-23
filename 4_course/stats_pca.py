"""
stats_pca.py — Метод головних компонент (МГК / PCA) для лабораторної роботи.

Алгоритм:
  1. Центрування та стандартизація даних (μ=0, σ=1)
  2. Кореляційна матриця R початкових стандартизованих даних
  3. Власні числа λ та власні вектори V матриці R
  4. Сортування власних значень за спаданням
  5. Обчислення часток поясненої та накопиченої дисперсії
  6. Відбір головних компонент, які покривають >95% загальної дисперсії (або заданий поріг)
  7. Координати в системі ГК: Z = X_std @ V
  8. Зворотне перетворення (відновлення): X_rec = Z_k @ V_k.T * σ + μ
  9. Розрахунок метрик похибки та якості відновлення (MSE, RMSE, MAE, R², r)
"""

import numpy as np
import pandas as pd
from scipy import stats


def pca_analysis(df: pd.DataFrame, selected_cols: list,
                 variance_threshold: float = 95.0,
                 k_components: int = None,
                 alpha: float = 0.05) -> dict:
    """
    Повне МГК-перетворення, відбір компонент та зворотне відновлення.

    Параметри
    ----------
    df                 : вхідний DataFrame
    selected_cols      : ознаки, які включаються в аналіз
    variance_threshold : поріг сумарної дисперсії у % (за замовчуванням 95.0)
    k_components       : точна кількість ГК для відбору (якщо None, визначається за variance_threshold)
    alpha              : рівень значущості

    Повертає словник з усіма проміжними та кінцевими результатами.
    """
    X = df[selected_cols].values.astype(float)
    n, p = X.shape

    # ── 1. Центрування та стандартизація ──────────────────────
    X_mean     = X.mean(axis=0)
    X_std_vals = X.std(axis=0, ddof=1)
    X_std_vals = np.where(X_std_vals == 0, 1.0, X_std_vals)   # захист від ділення на 0
    X_standardized = (X - X_mean) / X_std_vals

    # ── 2. Кореляційна матриця (на стандартизованих даних) ────
    R_orig = np.corrcoef(X_standardized.T)
    if p == 1:
        R_orig = np.array([[1.0]])

    # ── 3. Власні числа та власні вектори ─────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(R_orig)

    # Сортуємо за спаданням λ
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Стандартизація орієнтації векторів (максимальний за модулем елемент додатній)
    for j in range(p):
        max_idx = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[max_idx, j] < 0:
            eigenvectors[:, j] *= -1.0

    # ── 4. Частки дисперсії ───────────────────────────────────
    total_var = eigenvalues.sum()
    var_pct   = (eigenvalues / total_var) * 100.0
    cum_pct   = np.cumsum(var_pct)

    # Критерій Кайзера (кількість λ ≥ 1)
    kaiser_count = int(np.sum(eigenvalues >= 1.0))

    # ── 5. Відбір компонент, які покривають >95% дисперсії ─────
    if k_components is not None and 1 <= k_components <= p:
        k_retained = int(k_components)
    else:
        # Шукаємо перше k, де накопичений відсоток >= variance_threshold
        reached = np.where(cum_pct >= (variance_threshold - 1e-7))[0]
        if len(reached) > 0:
            k_retained = int(reached[0]) + 1
        else:
            k_retained = p

    # ── 6. Координати в системі МГК (всі та відібрані) ────────
    Z = X_standardized @ eigenvectors               # (n × p)
    Z_retained = Z[:, :k_retained]                  # (n × k_retained)
    V_retained = eigenvectors[:, :k_retained]        # (p × k_retained)

    # ── 7. Зворотне перетворення (відновлення) ────────────────
    # З відібраних k компонент (>95% дисперсії):
    X_rec_std = Z_retained @ V_retained.T
    X_rec     = X_rec_std * X_std_vals + X_mean

    # З усіх p компонент (точне 100% відновлення):
    X_rec_full_std = Z @ eigenvectors.T
    X_rec_full     = X_rec_full_std * X_std_vals + X_mean

    # ── 8. Оцінка точності відновлення (похибки) ──────────────
    diff = X - X_rec
    mse  = np.mean(diff ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(diff), axis=0)

    ss_tot = np.sum((X - X_mean) ** 2, axis=0)
    ss_res = np.sum(diff ** 2, axis=0)
    r2     = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), 1.0)

    r_corr = []
    for j in range(p):
        std_rec_j = X_rec[:, j].std(ddof=1)
        if std_rec_j > 1e-12 and X_std_vals[j] > 1e-12:
            r_val, _ = stats.pearsonr(X[:, j], X_rec[:, j])
        else:
            r_val = 1.0
        r_corr.append(r_val)
    r_corr = np.array(r_corr)

    retained_var = float(cum_pct[k_retained - 1])
    lost_var     = float(100.0 - retained_var)
    mean_mse     = float(np.mean(mse))

    # ── Кореляційні матриці ───────────────────────────────────
    R_pca = np.corrcoef(Z.T) if p > 1 else np.array([[1.0]])
    R_rec = np.corrcoef(X_rec.T) if p > 1 else np.array([[1.0]])

    # ── Центри розсіювання ────────────────────────────────────
    center_orig = X_mean.copy()
    center_pca  = Z.mean(axis=0)            # ≈ 0
    center_rec  = X_rec.mean(axis=0)        # ≈ X_mean

    # ── Назви осей та DataFrames ──────────────────────────────
    pc_names = [f"PC{i+1}" for i in range(p)]
    pc_names_with_pct = [f"PC{i+1} ({var_pct[i]:.1f}%)" for i in range(p)]
    pc_retained_names = pc_names_with_pct[:k_retained]

    df_orig = pd.DataFrame(X, columns=selected_cols)
    df_pca_all = pd.DataFrame(Z, columns=pc_names)
    df_pca_retained = pd.DataFrame(Z_retained, columns=pc_retained_names)
    df_rec = pd.DataFrame(X_rec, columns=[f"{col} (відновл.)" for col in selected_cols])

    # ── Таблиця МГК (формат Таблиці 4.5 посібника) ───────────
    table_rows = []
    # Рядки компонент власних векторів (факторних навантажень)
    for i, col in enumerate(selected_cols):
        row = {'Ознака': col}
        for j in range(p):
            row[pc_names[j]] = f"{eigenvectors[i, j]:+.4f}"
        table_rows.append(row)

    # Додаткові підсумкові рядки
    ev_row   = {'Ознака': 'Власні значення (λ)'}
    pct_row  = {'Ознака': 'Частка дисперсії (%)'}
    cum_row  = {'Ознака': 'Накопичена частка (%)'}
    kais_row = {'Ознака': 'Критерій Кайзера (λ≥1)'}

    for j in range(p):
        ev_row[pc_names[j]]   = f"{eigenvalues[j]:.4f}"
        pct_row[pc_names[j]]  = f"{var_pct[j]:.2f}%"
        cum_row[pc_names[j]]  = f"{cum_pct[j]:.2f}%"
        kais_row[pc_names[j]] = "✔ Так" if eigenvalues[j] >= 1.0 else "—"

    table_rows += [ev_row, pct_row, cum_row, kais_row]

    # ── Таблиця метрик зворотного перетворення ────────────────
    reconstruction_rows = []
    for j, col in enumerate(selected_cols):
        reconstruction_rows.append({
            'Ознака': col,
            'Сер. ориг.': f"{X_mean[j]:+.4f}",
            'Сер. відновл.': f"{X_rec[:, j].mean():+.4f}",
            'СКВ ориг.': f"{X_std_vals[j]:.4f}",
            'СКВ відновл.': f"{X_rec[:, j].std(ddof=1):.4f}",
            'MSE': f"{mse[j]:.6f}",
            'RMSE': f"{rmse[j]:.6f}",
            'MAE': f"{mae[j]:.6f}",
            'R²': f"{r2[j]:.4f}",
            'r(X, X̂)': f"{r_corr[j]:.4f}",
        })

    return {
        # ── Розмірності та вхідні дані ───────────────────────
        'n': n,
        'p': p,
        'cols': selected_cols,
        'alpha': alpha,
        'variance_threshold': variance_threshold,
        'k_retained': k_retained,
        'kaiser_count': kaiser_count,

        # ── Власні числа, вектори, дисперсії ─────────────────
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'var_pct': var_pct,
        'cum_pct': cum_pct,
        'retained_var_pct': retained_var,
        'lost_var_pct': lost_var,

        # ── Масиви координат ─────────────────────────────────
        'X_orig': X,
        'X_standardized': X_standardized,
        'Z': Z,
        'Z_retained': Z_retained,
        'V_retained': V_retained,
        'X_reconstructed': X_rec,
        'X_reconstructed_full': X_rec_full,

        # ── DataFrames ───────────────────────────────────────
        'df_orig': df_orig,
        'df_pca_all': df_pca_all,
        'df_pca_retained': df_pca_retained,
        'df_reconstructed': df_rec,

        # ── Назви осей ───────────────────────────────────────
        'pc_names': pc_names,
        'pc_names_with_pct': pc_names_with_pct,
        'pc_retained_names': pc_retained_names,

        # ── Метрики похибок ──────────────────────────────────
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'r_corr': r_corr,
        'mean_mse': mean_mse,

        # ── Таблиці ──────────────────────────────────────────
        'table_rows': table_rows,
        'reconstruction_rows': reconstruction_rows,

        # ── Кореляційні матриці ───────────────────────────────
        'R_orig': R_orig,
        'R_pca': R_pca,
        'R_rec': R_rec,

        # ── Центри ───────────────────────────────────────────
        'center_orig': center_orig,
        'center_pca': center_pca,
        'center_rec': center_rec,

        # ── Параметри стандартизації ─────────────────────────
        'X_mean': X_mean,
        'X_std_vals': X_std_vals,
    }

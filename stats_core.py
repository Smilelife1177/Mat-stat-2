"""
stats_core.py — Обчислювальне ядро: всі статистичні функції Лаб. 5
"""

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05  # рівень значущості за замовчуванням


# ──────────────────────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ ДАНИХ
# ──────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Завантажує дані з текстового файлу. Підтримує пробіли/таби/коми."""
    try:
        df = pd.read_csv(filepath, sep=r'\s+|,', engine='python', header=0)
        # Перевіряємо чи перший рядок числовий
        try:
            df.iloc[0].astype(float)
        except (ValueError, TypeError):
            pass
        else:
            df = pd.read_csv(filepath, sep=r'\s+|,', engine='python', header=None)
            df.columns = [f'X{i+1}' for i in range(df.shape[1])]
    except Exception:
        df = pd.read_csv(filepath, sep=r'\s+|,', engine='python', header=None)
        df.columns = [f'X{i+1}' for i in range(df.shape[1])]

    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df


# ──────────────────────────────────────────────────────────────
# 1. ПЕРВИННИЙ АНАЛІЗ
# ──────────────────────────────────────────────────────────────
def primary_stats(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Вектори середніх, СКВ, довірчі інтервали, мін/макс."""
    n = len(df)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    means = df.mean()
    stds  = df.std(ddof=1)
    se    = stds / np.sqrt(n)

    result = pd.DataFrame({
        'Середнє (x̄)':        means,
        'СКВ (s)':             stds,
        'SE':                  se,
        'ДІ нижня':            means - t_crit * se,
        'ДІ верхня':           means + t_crit * se,
        'Мін':                 df.min(),
        'Макс':                df.max(),
        'Медіана':             df.median(),
        'Асиметрія':           df.skew(),
        'Ексцес':              df.kurt(),
    }).round(5)
    result.attrs['t_crit']  = round(t_crit, 4)
    result.attrs['alpha']   = alpha
    result.attrs['n']       = n
    return result


# ──────────────────────────────────────────────────────────────
# 2. ПАРНІ КОЕФІЦІЄНТИ КОРЕЛЯЦІЇ
# ──────────────────────────────────────────────────────────────
def pearson_matrix(df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """
    Повертає словник:
      R         – матриця всіх парних кореляцій (numpy)
      P         – матриця p-values
      R_sig     – матриця тільки значущих кореляцій (решта = 0)
      r_crit    – критичне значення
      pairs     – список рядків деталей по кожній парі
      cols      – назви колонок
    """
    n, p = df.shape
    cols  = list(df.columns)
    R = np.zeros((p, p))
    P = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            r, pv = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
            R[i, j] = r
            P[i, j] = pv

    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    r_crit = t_crit / np.sqrt(n - 2 + t_crit ** 2)

    R_sig = R.copy()
    R_sig[np.abs(R) <= r_crit] = 0.0
    np.fill_diagonal(R_sig, 1.0)

    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            r  = R[i, j]
            pv = P[i, j]
            t_s = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-15))
            sig = "✔ Так" if pv < alpha else "✘ Ні"
            pairs.append({
                'Пара':       f"{cols[i]} — {cols[j]}",
                'r':          round(r, 5),
                't-статист.': round(t_s, 4),
                'p-value':    round(pv, 5),
                'Значущий':   sig,
            })

    return dict(R=R, P=P, R_sig=R_sig, r_crit=round(r_crit, 5),
                t_crit=round(t_crit, 4), pairs=pairs, cols=cols, n=n, alpha=alpha)


# ──────────────────────────────────────────────────────────────
# 3. ЧАСТКОВІ КОЕФІЦІЄНТИ КОРЕЛЯЦІЇ
# ──────────────────────────────────────────────────────────────
def partial_correlations(df: pd.DataFrame, R: np.ndarray,
                          alpha: float = ALPHA) -> dict:
    """
    Часткові кореляції через precision matrix (обернена R).
    r_ij.rest = -R_inv[i,j] / sqrt(R_inv[i,i]*R_inv[j,j])
    """
    n, p = df.shape
    cols = list(df.columns)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return dict(error="Матриця кореляцій вироджена")

    PR = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            denom = np.sqrt(R_inv[i, i] * R_inv[j, j])
            PR[i, j] = -R_inv[i, j] / denom if denom > 1e-15 else 0.0
    np.fill_diagonal(PR, 1.0)

    df_deg = n - p   # ступені свободи
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_deg)

    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            r  = PR[i, j]
            t_s = r * np.sqrt(df_deg) / np.sqrt(max(1 - r**2, 1e-15))
            pv  = 2 * (1 - stats.t.cdf(abs(t_s), df=df_deg))
            sig = pv < alpha

            ci_lo = ci_hi = float('nan')
            if sig and abs(r) < 1.0:
                z      = np.arctanh(r)
                se_z   = 1.0 / np.sqrt(max(df_deg - 1, 1))
                z_crit = stats.norm.ppf(1 - alpha / 2)
                ci_lo  = np.tanh(z - z_crit * se_z)
                ci_hi  = np.tanh(z + z_crit * se_z)

            pairs.append({
                'Пара':       f"{cols[i]} — {cols[j]}",
                'r_part':     round(r, 5),
                't-статист.': round(t_s, 4),
                'p-value':    round(pv, 5),
                'Значущий':   "✔ Так" if sig else "✘ Ні",
                'ДІ нижня':   round(ci_lo, 5) if not np.isnan(ci_lo) else "—",
                'ДІ верхня':  round(ci_hi, 5) if not np.isnan(ci_hi) else "—",
            })

    return dict(PR=PR, pairs=pairs, cols=cols,
                df_deg=df_deg, t_crit=round(t_crit, 4), alpha=alpha)


# ──────────────────────────────────────────────────────────────
# 4. МНОЖИННІ КОЕФІЦІЄНТИ КОРЕЛЯЦІЇ
# ──────────────────────────────────────────────────────────────
def multiple_correlations(df: pd.DataFrame, R: np.ndarray,
                           alpha: float = ALPHA) -> dict:
    """
    R²_i = 1 - 1/R_inv[i,i]   (через precision matrix)
    F-тест значущості: F = (R²/(p-1)) / ((1-R²)/(n-p))
    """
    n, p = df.shape
    cols  = list(df.columns)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return dict(error="Матриця кореляцій вироджена")

    df1    = p - 1
    df2    = n - p
    F_crit = stats.f.ppf(1 - alpha, dfn=df1, dfd=df2)

    rows = []
    for i in range(p):
        R2  = max(1.0 - 1.0 / R_inv[i, i], 0.0)
        Rm  = np.sqrt(R2)
        F   = (R2 / df1) / ((1 - R2) / df2) if R2 < 1.0 else float('inf')
        pv  = 1 - stats.f.cdf(F, dfn=df1, dfd=df2)
        rows.append({
            'Змінна':     cols[i],
            'R_mult':     round(Rm, 5),
            'R²':         round(R2, 5),
            'F-статист.': round(F, 4),
            'p-value':    round(pv, 5),
            'Значущий':   "✔ Так" if pv < alpha else "✘ Ні",
        })

    return dict(rows=rows, df1=df1, df2=df2,
                F_crit=round(F_crit, 4), alpha=alpha)
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
def _partial_r_recursive(R_mat: np.ndarray, i: int, j: int,
                          cond: list) -> float:
    """
    Рекурентна формула посібника (розд. 4.3):
      r_ij·cd = (r_ij·c − r_id·c · r_jd·c) /
                sqrt((1 − r²_id·c)(1 − r²_jd·c))
    Базовий випадок (cond=[]):  повертає R_mat[i,j].
    """
    if not cond:
        return R_mat[i, j]
    # Беремо перший елемент з набору c як змінну d
    d    = cond[0]
    rest = cond[1:]
    r_ij = _partial_r_recursive(R_mat, i, j, rest)
    r_id = _partial_r_recursive(R_mat, i, d, rest)
    r_jd = _partial_r_recursive(R_mat, j, d, rest)
    denom = np.sqrt(max(1 - r_id**2, 1e-15) * max(1 - r_jd**2, 1e-15))
    return (r_ij - r_id * r_jd) / denom


def partial_correlations(df: pd.DataFrame, R: np.ndarray,
                          alpha: float = ALPHA) -> dict:
    """
    Часткові коефіцієнти кореляції за рекурентною формулою посібника
    (розділ 4.3, формула r_ij·d).

    Для кожної пари (i,j) виключаємо ВСІ інші p-2 змінних (повний порядок).
    w = p - 2  — кількість виключених змінних.

    t-тест:  t = r_ij·c · sqrt(N - w - 2) / sqrt(1 - r²_ij·c)
             df = N - w - 2  (посібник, розд. 4.3)

    ДІ (перетворення Фішера, посібник):
      v1 = 0.5·ln((1+r)/(1-r)) − u_α/2 / sqrt(N - w - 3)
      v2 = 0.5·ln((1+r)/(1-r)) + u_α/2 / sqrt(N - w - 3)
      ДІ: [(exp(2v1)-1)/(exp(2v1)+1), (exp(2v2)-1)/(exp(2v2)+1)]
    """
    N, p = df.shape
    cols  = list(df.columns)
    w     = p - 2          # кількість виключених змінних (повний порядок)
    df_t  = N - w - 2      # ступені свободи t-розподілу
    t_crit = stats.t.ppf(1 - alpha / 2, df=max(df_t, 1))

    # Матриця часткових кореляцій
    PR = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                cond = [k for k in range(p) if k != i and k != j]
                PR[i, j] = _partial_r_recursive(R, i, j, cond)

    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            r   = PR[i, j]
            t_s = r * np.sqrt(max(df_t, 1)) / np.sqrt(max(1 - r**2, 1e-15))
            pv  = 2 * (1 - stats.t.cdf(abs(t_s), df=max(df_t, 1)))
            sig = pv < alpha

            # ДІ — формула посібника (розд. 4.3)
            ci_lo = ci_hi = float('nan')
            if sig and abs(r) < 1.0 - 1e-9:
                df_ci  = max(N - w - 3, 1)
                z      = 0.5 * np.log((1 + r) / (1 - r))   # arctanh(r)
                u_crit = stats.norm.ppf(1 - alpha / 2)
                delta  = u_crit / np.sqrt(df_ci)
                v1, v2 = z - delta, z + delta
                ci_lo  = (np.exp(2 * v1) - 1) / (np.exp(2 * v1) + 1)
                ci_hi  = (np.exp(2 * v2) - 1) / (np.exp(2 * v2) + 1)

            pairs.append({
                'Пара':       f"{cols[i]} — {cols[j]}",
                'r_part':     round(r, 5),
                't-статист.': round(t_s, 4),
                'p-value':    round(pv, 5),
                'Значущий':   "✔ Так" if sig else "✘ Ні",
                'ДІ нижня':   round(ci_lo, 5) if not np.isnan(ci_lo) else "—",
                'ДІ верхня':  round(ci_hi, 5) if not np.isnan(ci_hi) else "—",
            })

    return dict(PR=PR, pairs=pairs, cols=cols, w=w,
                df_deg=df_t, t_crit=round(t_crit, 4), alpha=alpha)


# ──────────────────────────────────────────────────────────────
# 4. МНОЖИННІ КОЕФІЦІЄНТИ КОРЕЛЯЦІЇ
# ──────────────────────────────────────────────────────────────
def multiple_correlations(df: pd.DataFrame, R: np.ndarray,
                           alpha: float = ALPHA) -> dict:
    """
    Формула посібника (розд. 4.3):
      R²_k = 1 − |R_kk| / |R|
    де |R_kk| — мінор матриці R (викреслити k-й рядок і k-й стовпець).

    F-тест (посібник, розд. 4.3):
      f = R²_k · (N − n − 1) / (n · (1 − R²_k))
      ~ F(ν1 = n−1, ν2 = N − n − 1)   [n = кількість ознак, N = обсяг]
    """
    N, p  = df.shape
    cols  = list(df.columns)
    det_R = np.linalg.det(R)

    # ν1 = p−1 (решта ознак), ν2 = N − p − 0  (посібник: N−n−1, n=кількість ознак)
    nu1    = p - 1
    nu2    = N - p - 1      # посібник: N − n − 1
    F_crit = stats.f.ppf(1 - alpha, dfn=nu1, dfd=max(nu2, 1))

    rows = []
    for k in range(p):
        # Мінор: видаляємо k-й рядок і k-й стовпець
        minor = np.delete(np.delete(R, k, axis=0), k, axis=1)
        det_minor = np.linalg.det(minor)

        if abs(det_minor) > 1e-15:
            R2 = max(1.0 - det_R / det_minor, 0.0)   # посібник: 1 − |R|/|R_kk|
        else:
            R2 = 0.0
        Rm = np.sqrt(R2)

        # F-тест (посібник: nu1 = n−1 = p−1, nu2 = N−n−1 = N−p−1)
        if R2 < 1.0 - 1e-12 and nu2 > 0:
            F  = (R2 * nu2) / ((1 - R2) * nu1)
        else:
            F  = float('inf')
        pv = 1 - stats.f.cdf(F, dfn=nu1, dfd=max(nu2, 1))

        rows.append({
            'Змінна':     cols[k],
            'R_mult':     round(Rm, 5),
            'R²':         round(R2, 5),
            'F-статист.': round(F, 4),
            'p-value':    round(pv, 5),
            'Значущий':   "✔ Так" if pv < alpha else "✘ Ні",
        })

    return dict(rows=rows, df1=nu1, df2=nu2,
                F_crit=round(F_crit, 4), alpha=alpha)
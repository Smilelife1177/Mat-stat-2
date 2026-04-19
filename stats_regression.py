"""
stats_regression.py — Множинна лінійна регресія (пункт 3)
"""
import numpy as np
import pandas as pd
from scipy import stats

def multiple_linear_regression(df: pd.DataFrame, dep_col: str, alpha: float = 0.05) -> dict:
    """Повна множинна лінійна регресія за вимогами викладача"""
    y = df[dep_col].values
    X = df.drop(columns=[dep_col]).values
    n, p = X.shape
    X = np.column_stack((np.ones(n), X))          # додаємо стовпець 1 (вільний член)
    k = p + 1                                      # кількість параметрів (включаючи b0)

    # Оцінки параметрів
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y

    # Залишки та дисперсія
    y_hat = X @ beta
    residuals = y - y_hat
    RSS = np.sum(residuals ** 2)
    MSE = RSS / (n - k)
    sigma = np.sqrt(MSE)

    # Стандартні помилки та t-статистики
    se_beta = np.sqrt(np.diag(XtX_inv) * MSE)
    t_stat = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
    t_crit = stats.t.ppf(1 - alpha/2, n - k)

    # Довірчі інтервали параметрів
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta

    # Стандартизовані коефіцієнти
    X_std = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0, ddof=1)
    beta_std = np.append(beta[0], beta[1:] * X[:, 1:].std(axis=0, ddof=1) / y.std(ddof=1))

    # Коефіцієнт детермінації
    TSS = np.sum((y - y.mean()) ** 2)
    R2 = 1 - RSS / TSS
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - k)

    # F-тест моделі
    F_stat = (R2 / p) / ((1 - R2) / (n - k))
    F_crit = stats.f.ppf(1 - alpha, p, n - k)
    model_sig = F_stat > F_crit

    # Толерантні межі для залишкової дисперсії (довірчий інтервал σ²)
    chi2_lo = stats.chi2.ppf(alpha/2, n - k)
    chi2_hi = stats.chi2.ppf(1 - alpha/2, n - k)
    sigma2_ci_lo = (n - k) * MSE / chi2_hi
    sigma2_ci_hi = (n - k) * MSE / chi2_lo

    # Діагностична таблиця
    coef_table = []
    names = ['b0 (вільний член)'] + [f'b{i}' for i in range(1, k)]
    for i in range(k):
        coef_table.append({
            'Параметр': names[i],
            'Оцінка β': round(beta[i], 5),
            'Стандартизований β': round(beta_std[i], 4),
            'SE': round(se_beta[i], 5),
            't-статист.': round(t_stat[i], 4),
            'p-value': round(p_values[i], 5),
            'Значущий': "✔ Так" if p_values[i] < alpha else "✘ Ні",
            'ДІ нижня': round(ci_lower[i], 5),
            'ДІ верхня': round(ci_upper[i], 5),
        })

    return {
        'coef_table': coef_table,
        'R2': round(R2, 5),
        'R2_adj': round(R2_adj, 5),
        'F_stat': round(F_stat, 4),
        'F_crit': round(F_crit, 4),
        'model_sig': model_sig,
        'sigma2_ci_lo': round(sigma2_ci_lo, 5),
        'sigma2_ci_hi': round(sigma2_ci_hi, 5),
        'residuals': residuals,
        'y_hat': y_hat,
        'y': y,
        'dep_col': dep_col,
        'predictors': list(df.drop(columns=[dep_col]).columns),
        'n': n,
        'k': k,
        'alpha': alpha
    }
"""Метод головних факторів (МГФ, principal-axis factoring) для Лаб-2."""

import numpy as np
import pandas as pd


def _clip(h):
    """Загальності мають належати [0.005; 0.995]."""
    return np.clip(np.nan_to_num(h, nan=.005, posinf=.995, neginf=.005), .005, .995)


def _reduced(R, h):
    out = R.copy()
    np.fill_diagonal(out, h)
    return out


def _solution(Rred, n_factors):
    """Власні корені редукційної матриці та не обернені навантаження."""
    vals, vecs = np.linalg.eigh(Rred)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # У методі головних факторів використовуються тільки додатні корені.
    k = min(n_factors, int(np.sum(vals > 1e-10)))
    if k == 0:
        return vals, np.empty((Rred.shape[0], 0))
    loadings = vecs[:, :k] * np.sqrt(vals[:k])
    # Стабільний знак стовпців для однакового виводу між запусками.
    for j in range(k):
        i = np.argmax(np.abs(loadings[:, j]))
        if loadings[i, j] < 0:
            loadings[:, j] *= -1
    return vals, loadings


def _initial_estimates(R):
    """Шість поширених стартових оцінок загальностей для МГФ."""
    p = R.shape[0]
    off_sq = R ** 2
    np.fill_diagonal(off_sq, 0)
    inv = np.linalg.pinv(R)
    smc = 1 - 1 / np.maximum(np.diag(inv), 1e-12)
    eigvals, eigvecs = np.linalg.eigh(R)
    lead = eigvecs[:, -1] * np.sqrt(max(eigvals[-1], 0))
    # Кожна оцінка прямо пояснена у UI: це не шість різних моделей.
    return {
        "1. Одинична": np.ones(p),
        "2. Макс. r²": np.max(off_sq, axis=1),
        "3. Середнє r²": np.sum(off_sq, axis=1) / max(p - 1, 1),
        "4. Сума r²": np.sum(off_sq, axis=1),
        "5. SMC": smc,
        "6. 1-й головний фактор": lead ** 2,
    }


def factor_analysis(df: pd.DataFrame, columns=None, n_factors=None,
                    tolerance=1e-4, max_iter=50):
    """Повний ітераційний МГФ з аудитом усіх проміжних матриць.

    Найкращою стартовою оцінкою вважається та, що дає найменший RMSR
    (RMS невідтворених позадіагональних кореляцій) до початку ітерацій.
    """
    data = df if columns is None else df[columns]
    X = data.apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[1] < 2 or X.shape[0] < 3:
        raise ValueError("Для МГФ потрібні щонайменше 3 спостереження і 2 змінні.")
    if (X.std(ddof=1) == 0).any():
        raise ValueError("МГФ неможливий: серед обраних змінних є стала ознака.")
    R = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    p = R.shape[0]
    raw_eigs = np.linalg.eigvalsh(R)[::-1]
    if n_factors is None:
        k = max(1, int(np.sum(raw_eigs > 1.0)))
    else:
        k = max(1, min(int(n_factors), p - 1))

    candidates = []
    for name, h in _initial_estimates(R).items():
        h = _clip(h)
        red = _reduced(R, h)
        vals, L = _solution(red, k)
        reproduced = L @ L.T
        mask = ~np.eye(p, dtype=bool)
        rmsr = float(np.sqrt(np.mean((R[mask] - reproduced[mask]) ** 2)))
        candidates.append({"name": name, "h": h, "reduced": red,
                           "eigenvalues": vals, "loadings": L, "rmsr": rmsr})
    best = min(candidates, key=lambda c: c["rmsr"])

    history = []
    h = best["h"].copy()
    for iteration in range(1, max_iter + 1):
        red = _reduced(R, h)
        vals, L = _solution(red, k)
        h_next = _clip(np.sum(L ** 2, axis=1))
        delta = float(np.max(np.abs(h_next - h)))
        history.append({"iteration": iteration, "h_before": h.copy(),
                        "reduced": red, "eigenvalues": vals,
                        "loadings": L, "h_after": h_next.copy(), "delta": delta})
        h = h_next
        if delta < tolerance:
            break

    final = history[-1]
    return {"columns": list(X.columns), "n": len(X), "R": R, "raw_eigenvalues": raw_eigs,
            "n_factors": k, "candidates": candidates, "selected": best,
            "history": history, "loadings": final["loadings"],
            "communalities": final["h_after"], "converged": final["delta"] < tolerance,
            "tolerance": tolerance, "max_iter": max_iter}

"""Інтерфейс Лаб-2: ітераційний метод головних факторів."""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

import stats_core as sc
from stats_factor import factor_analysis
from gui_widgets import C, DataTable, MetricCard, SectionHeader


def _matrix_rows(matrix, names, digits=4):
    return [[names[i]] + [f"{v:.{digits}f}" for v in row]
            for i, row in enumerate(matrix)]


class TabLab2(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app, self.df, self.result = app, None, None
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        top.pack(fill='x')
        tk.Label(top, text="Лабораторна робота №2 — ітераційний метод МГФ",
                 bg=C['bg'], fg=C['accent'], font=('Consolas', 12, 'bold')).pack(side='left')
        ttk.Button(top, text="▶ Розрахувати МГФ", style='Accent.TButton', command=self._run).pack(side='right')
        ttk.Button(top, text="Завантажити файл…", command=self._browse).pack(side='right', padx=6)
        self.path = tk.StringVar(value="MGF.DAT")
        tk.Entry(top, textvariable=self.path, width=20, bg=C['panel2'], fg=C['text'],
                 insertbackground=C['accent'], relief='flat').pack(side='right', padx=6)
        tk.Label(top, text="Дані:", bg=C['bg'], fg=C['text']).pack(side='right')

        opts = tk.Frame(self, bg=C['bg'], padx=16, pady=(0, 8))
        opts.pack(fill='x')
        tk.Label(opts, text="Кількість факторів (0 = критерій Кайзера):", bg=C['bg'], fg=C['subtext']).pack(side='left')
        self.k_var = tk.StringVar(value="0")
        ttk.Spinbox(opts, from_=0, to=20, textvariable=self.k_var, width=5).pack(side='left', padx=6)
        tk.Label(opts, text="ε:", bg=C['bg'], fg=C['subtext']).pack(side='left', padx=(14, 0))
        self.eps_var = tk.StringVar(value="0.0001")
        ttk.Entry(opts, textvariable=self.eps_var, width=9).pack(side='left', padx=6)
        tk.Label(opts, text="Усі шість стартових оцінок показані окремо; оптимум — мінімальний RMSR.",
                 bg=C['bg'], fg=C['subtext'], font=('Consolas', 8)).pack(side='left', padx=14)

        self.cards = tk.Frame(self, bg=C['bg']); self.cards.pack(fill='x', padx=16, pady=(0, 5))
        self.nb = ttk.Notebook(self); self.nb.pack(fill='both', expand=True, padx=8, pady=8)
        self.t_r = ttk.Frame(self.nb); self.t_start = ttk.Frame(self.nb); self.t_iter = ttk.Frame(self.nb); self.t_final = ttk.Frame(self.nb)
        self.nb.add(self.t_r, text="  Матриця R  ")
        self.nb.add(self.t_start, text="  6 оцінок і редукції  ")
        self.nb.add(self.t_iter, text="  Ітерації МГФ  ")
        self.nb.add(self.t_final, text="  Результат: навантаження  ")
        self._load_default()

    def _load_default(self):
        try:
            self.df = sc.load_data(self.path.get())
        except Exception:
            self.df = None

    def _browse(self):
        f = filedialog.askopenfilename(title="Дані для МГФ", filetypes=[("Data", "*.dat *.txt *.csv"), ("All", "*.*")])
        if f:
            self.path.set(f); self.df = None

    def _data(self):
        path = self.path.get().strip()
        if path and os.path.exists(path):
            return sc.load_data(path)
        if self.app.df is not None:
            return self.app.df
        raise ValueError("Не знайдено файл MGF.DAT і не завантажено інші дані.")

    @staticmethod
    def _clear(frame):
        for w in frame.winfo_children(): w.destroy()

    def _table(self, parent, matrix, columns, heading="Змінна", row_names=None):
        row_names = columns if row_names is None else row_names
        DataTable(parent, [heading] + columns, _matrix_rows(matrix, row_names)).pack(fill='both', expand=True, padx=5, pady=5)

    def _run(self):
        try:
            df = self._data()
            k = int(self.k_var.get()); eps = float(self.eps_var.get())
            self.result = factor_analysis(df, n_factors=None if k == 0 else k, tolerance=eps)
            self.df = df
            self._render()
            self.app.status.ok("Лаб-2: МГФ розраховано")
        except Exception as e:
            messagebox.showerror("Лаб-2 — помилка", str(e)); self.app.status.err(str(e))

    def _render(self):
        r, names = self.result, self.result['columns']
        for w in self.cards.winfo_children(): w.destroy()
        MetricCard(self.cards, "Спостережень × змінних", f"{r['n']} × {len(names)}").pack(side='left', padx=4)
        MetricCard(self.cards, "Факторів", r['n_factors'], C['green']).pack(side='left', padx=4)
        MetricCard(self.cards, "Ітерацій", len(r['history']), C['yellow']).pack(side='left', padx=4)
        MetricCard(self.cards, "Стан", "збіжність" if r['converged'] else "ліміт ітерацій", C['green'] if r['converged'] else C['accent2']).pack(side='left', padx=4)

        self._clear(self.t_r); self._table(self.t_r, r['R'], names, "R")
        self._clear(self.t_start)
        inner = ttk.Notebook(self.t_start); inner.pack(fill='both', expand=True)
        for c in r['candidates']:
            page = ttk.Frame(inner); inner.add(page, text=f"  {c['name']}  ")
            txt = tk.Label(page, text=f"h² = " + ", ".join(f"{x:.4f}" for x in c['h']) + f"     RMSR = {c['rmsr']:.6f}",
                           bg=C['bg'], fg=C['green'] if c is r['selected'] else C['subtext'], font=('Consolas', 9))
            txt.pack(anchor='w', padx=8, pady=6)
            self._table(page, c['reduced'], names, "R*")

        self._clear(self.t_iter)
        itnb = ttk.Notebook(self.t_iter); itnb.pack(fill='both', expand=True)
        for item in r['history']:
            page = ttk.Frame(itnb); itnb.add(page, text=f"  Ітерація {item['iteration']}  ")
            tk.Label(page, text=f"max|Δh²| = {item['delta']:.6g}     h² після: " + ", ".join(f"{x:.4f}" for x in item['h_after']),
                     bg=C['bg'], fg=C['subtext'], font=('Consolas', 8)).pack(anchor='w', padx=8, pady=5)
            split = ttk.PanedWindow(page, orient='horizontal'); split.pack(fill='both', expand=True, padx=5, pady=3)
            left, right = ttk.Frame(split), ttk.Frame(split); split.add(left, weight=1); split.add(right, weight=1)
            SectionHeader(left, "Редукційна матриця R*").pack(fill='x'); self._table(left, item['reduced'], names, "R*")
            SectionHeader(right, "Матриця навантажень A").pack(fill='x')
            factors = [f"F{i+1}" for i in range(item['loadings'].shape[1])]
            self._table(right, item['loadings'], factors, "Змінна", names)

        self._clear(self.t_final)
        chosen = r['selected']
        tk.Label(self.t_final, text=f"Обрана початкова оцінка: {chosen['name']} (найменший RMSR = {chosen['rmsr']:.6f}).",
                 bg=C['bg'], fg=C['green'], font=('Consolas', 10, 'bold')).pack(anchor='w', padx=10, pady=(10, 4))
        tk.Label(self.t_final, text="Підсумкова матриця факторних навантажень A; h² — сума квадратів навантажень у рядку.",
                 bg=C['bg'], fg=C['subtext'], font=('Consolas', 9)).pack(anchor='w', padx=10, pady=(0, 4))
        factors = [f"F{i+1}" for i in range(r['loadings'].shape[1])]
        rows = [[names[i]] + [f"{x:.5f}" for x in r['loadings'][i]] + [f"{r['communalities'][i]:.5f}"] for i in range(len(names))]
        DataTable(self.t_final, ["Змінна"] + factors + ["h²"], rows).pack(fill='both', expand=True, padx=8, pady=8)

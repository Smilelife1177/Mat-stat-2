"""
main.py — Головний файл GUI. Лабораторна робота №5.
Запуск: python main.py

Структура файлів:
  main.py         — цей файл (головне вікно, вкладки)
  stats_core.py   — всі статистичні обчислення
  plots.py        — всі matplotlib-графіки
  gui_widgets.py  — перевикористовувані tkinter-компоненти
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import numpy as np
import pandas as pd

import stats_core as sc
import plots
from gui_widgets import (
    apply_theme, MetricCard, DataTable,
    PlotPanel, SectionHeader, StatusBar, C
)
from tab_theory import TabTheory
import stats_regression as sr
import plots_regression as pr


# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА 0 — ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════════════════════════
class TabLoad(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        # Верхня панель вибору файлу
        top = tk.Frame(self, bg=C['bg'], pady=14, padx=16)
        top.pack(fill='x')

        tk.Label(top, text="Лабораторна робота №5",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 18, 'bold')).pack(anchor='w')
        tk.Label(top, text="Аналіз багатовимірних об'єктів спостережень",
                 bg=C['bg'], fg=C['subtext'],
                 font=('Consolas', 10)).pack(anchor='w', pady=(2, 10))

        file_row = tk.Frame(top, bg=C['bg'])
        file_row.pack(fill='x')
        tk.Label(file_row, text="Файл даних:", bg=C['bg'],
                 fg=C['text'], font=('Consolas', 9)).pack(side='left')
        self.path_var = tk.StringVar(value="norm3n.txt")
        ent = tk.Entry(file_row, textvariable=self.path_var,
                       bg=C['panel2'], fg=C['text'],
                       insertbackground=C['accent'],
                       font=('Consolas', 9), relief='flat',
                       bd=1, width=46)
        ent.pack(side='left', padx=(8, 6))
        ttk.Button(file_row, text="Огляд…",
                   command=self._browse).pack(side='left', padx=2)
        ttk.Button(file_row, text="▶  Завантажити",
                   style='Accent.TButton',
                   command=self._load).pack(side='left', padx=(8, 0))

        # Параметри
        params = tk.Frame(top, bg=C['bg'], pady=8)
        params.pack(fill='x')
        tk.Label(params, text="Рівень значущості α:", bg=C['bg'],
                 fg=C['text'], font=('Consolas', 9)).pack(side='left')
        self.alpha_var = tk.StringVar(value="0.05")
        alpha_cb = ttk.Combobox(params, textvariable=self.alpha_var,
                                values=['0.01', '0.05', '0.10'],
                                width=6, state='readonly')
        alpha_cb.pack(side='left', padx=(6, 0))

        # Превью даних
        ttk.Separator(self).pack(fill='x', pady=4)
        SectionHeader(self, "  Попередній перегляд даних").pack(
            fill='x', padx=16, pady=(6, 2))

        prev_frame = ttk.Frame(self, padding=10)
        prev_frame.pack(fill='both', expand=True)

        # Таблиця превью
        self.preview_table = DataTable(prev_frame, columns=['—'], data=[])
        self.preview_table.pack(fill='both', expand=True)

        # Картки метрик
        self.cards_frame = tk.Frame(self, bg=C['bg'])
        self.cards_frame.pack(fill='x', padx=16, pady=8)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Відкрити файл даних",
            filetypes=[("Text files", "*.txt *.csv *.dat"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _load(self):
        path  = self.path_var.get().strip()
        alpha = float(self.alpha_var.get())
        if not os.path.exists(path):
            messagebox.showerror("Помилка", f"Файл не знайдено:\n{path}")
            self.app.status.err(f"Файл не знайдено: {path}")
            return

        try:
            df = sc.load_data(path)
        except Exception as e:
            messagebox.showerror("Помилка зчитування", str(e))
            self.app.status.err(str(e))
            return

        self.app.df    = df
        self.app.alpha = alpha
        self._show_preview(df)
        
        # Оновити Combobox у вкладці регресії
        for tab in self.app.notebook.tabs():
            widget = self.app.notebook.nametowidget(tab)
            if isinstance(widget, TabRegression):
                widget.update_columns(df)
        
        self.app.status.ok(
            f"Завантажено: {df.shape[0]} спостережень × {df.shape[1]} ознак  |  "
            f"Файл: {os.path.basename(path)}")
        self.app.notebook.select(1)   # перейти до наступної вкладки

    def _show_preview(self, df):
        # Оновити таблицю превью
        cols = list(df.columns)
        self.preview_table.destroy()
        table = DataTable(self.preview_table.master,
                          columns=cols,
                          data=df.head(30).values.tolist())
        table.pack(fill='both', expand=True)
        self.preview_table = table

        # Картки метрик
        for w in self.cards_frame.winfo_children():
            w.destroy()
        MetricCard(self.cards_frame, "Спостережень", df.shape[0],
                   color=C['green']).pack(side='left', padx=(0, 8))
        MetricCard(self.cards_frame, "Ознак", df.shape[1],
                   color=C['accent']).pack(side='left', padx=(0, 8))
        MetricCard(self.cards_frame, "Пропущених", int(df.isna().sum().sum()),
                   color=C['accent2']).pack(side='left', padx=(0, 8))
        for col in df.columns:
            MetricCard(self.cards_frame, col,
                       f"μ={df[col].mean():.3f}  σ={df[col].std():.3f}",
                       color=C['yellow']).pack(side='left', padx=(0, 6))


# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА 1 — ПЕРВИННИЙ АНАЛІЗ
# ══════════════════════════════════════════════════════════════
class TabPrimary(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        # Кнопка запуску
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')
        tk.Label(ctrl, text="Первинний статистичний аналіз",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 12, 'bold')).pack(side='left')
        ttk.Button(ctrl, text="▶  Розрахувати", style='Accent.TButton',
                   command=self._run).pack(side='right')

        ttk.Separator(self).pack(fill='x')

        # Paned: ліворуч таблиця, праворуч гістограми
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        SectionHeader(left, "  Зведена таблиця", accent=True).pack(
            fill='x', padx=6, pady=(6, 2))
        self.table_frame = ttk.Frame(left)
        self.table_frame.pack(fill='both', expand=True, padx=6, pady=4)

        SectionHeader(right, "  Гістограми з кривою нормального розподілу",
                      accent=True).pack(fill='x', padx=6, pady=(6, 2))
        self.plot_panel = PlotPanel(right)
        self.plot_panel.pack(fill='both', expand=True, padx=4)

    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані (вкладка «Дані»)")
            return
        self.app.status.info("Обчислення первинного аналізу…")
        result = sc.primary_stats(self.app.df, self.app.alpha)

        # Таблиця
        for w in self.table_frame.winfo_children():
            w.destroy()
        cols = ['Ознака'] + list(result.columns)
        data = []
        for idx, row in result.iterrows():
            data.append([idx] + [round(v, 4) if isinstance(v, float) else v
                                 for v in row.values])
        DataTable(self.table_frame, columns=cols, data=data).pack(
            fill='both', expand=True)

        # Гістограми
        fig = plots.histograms_figure(self.app.df)
        self.plot_panel.show(fig)
        self.app.status.ok(
            f"Первинний аналіз: t_крит={result.attrs['t_crit']}  "
            f"α={result.attrs['alpha']}")


# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА 2 — ПАРНІ КОРЕЛЯЦІЇ
# ══════════════════════════════════════════════════════════════
class TabPairCorr(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._res = None
        self._build()

    def _build(self):
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')
        tk.Label(ctrl, text="Матриця парних коефіцієнтів кореляції (Пірсон)",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 12, 'bold')).pack(side='left')
        ttk.Button(ctrl, text="▶  Розрахувати", style='Accent.TButton',
                   command=self._run).pack(side='right')
        ttk.Separator(self).pack(fill='x')

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        # Вкладки всередині
        self.tab_heatmap_all = ttk.Frame(nb)
        self.tab_heatmap_sig = ttk.Frame(nb)
        self.tab_table       = ttk.Frame(nb)
        self.tab_scatter     = ttk.Frame(nb)
        nb.add(self.tab_heatmap_all, text="  Теплова карта (всі)  ")
        nb.add(self.tab_heatmap_sig, text="  Теплова карта (значущі)  ")
        nb.add(self.tab_table,       text="  Деталі по парах  ")
        nb.add(self.tab_scatter,     text="  Scatter-матриця  ")

        self.pp_all  = PlotPanel(self.tab_heatmap_all)
        self.pp_all.pack(fill='both', expand=True)
        self.pp_sig  = PlotPanel(self.tab_heatmap_sig)
        self.pp_sig.pack(fill='both', expand=True)
        self.pp_scat = PlotPanel(self.tab_scatter)
        self.pp_scat.pack(fill='both', expand=True)

        self.table_fr = ttk.Frame(self.tab_table)
        self.table_fr.pack(fill='both', expand=True, padx=6, pady=6)

        # Мета-інфо
        self.meta_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.meta_var, bg=C['bg'],
                 fg=C['subtext'], font=('Consolas', 8)).pack(
            anchor='w', padx=16, pady=(0, 4))

    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані")
            return
        self.app.status.info("Обчислення парних кореляцій…")

        res = sc.pearson_matrix(self.app.df, self.app.alpha)
        self._res = res
        self.app._pair_corr_res = res  # зберігаємо для наступних вкладок

        cols = res['cols']

        # Теплові карти
        self.pp_all.show(plots.heatmap_figure(
            res['R'], cols, "Матриця всіх парних коефіцієнтів кореляції"))
        self.pp_sig.show(plots.heatmap_figure(
            res['R_sig'], cols,
            f"Матриця ЗНАЧУЩИХ парних коефіцієнтів кореляції  (|r| > {res['r_crit']})"))
        self.pp_scat.show(plots.scatter_matrix_figure(self.app.df))

        # Таблиця
        for w in self.table_fr.winfo_children():
            w.destroy()
        if res['pairs']:
            table_cols = list(res['pairs'][0].keys())
            DataTable(self.table_fr, columns=table_cols,
                      data=res['pairs']).pack(fill='both', expand=True)

        self.meta_var.set(
            f"n={res['n']}   α={res['alpha']}   "
            f"|r_крит|={res['r_crit']}   t_крит={res['t_crit']}")
        self.app.status.ok(
            f"Парні кореляції: r_крит={res['r_crit']}  α={res['alpha']}")


# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА 3 — ЧАСТКОВІ КОРЕЛЯЦІЇ
# ══════════════════════════════════════════════════════════════
class TabPartial(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')
        tk.Label(ctrl, text="Часткові коефіцієнти кореляції",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 12, 'bold')).pack(side='left')
        ttk.Button(ctrl, text="▶  Розрахувати", style='Accent.TButton',
                   command=self._run).pack(side='right')
        ttk.Separator(self).pack(fill='x')

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        left  = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        # Ліворуч: вкладки таблиця/heatmap
        nb = ttk.Notebook(left)
        nb.pack(fill='both', expand=True)
        self.tab_tbl  = ttk.Frame(nb)
        self.tab_heat = ttk.Frame(nb)
        nb.add(self.tab_tbl,  text="  Таблиця деталей  ")
        nb.add(self.tab_heat, text="  Теплова карта  ")
        self.tbl_fr = ttk.Frame(self.tab_tbl)
        self.tbl_fr.pack(fill='both', expand=True, padx=4, pady=4)
        self.pp_heat = PlotPanel(self.tab_heat)
        self.pp_heat.pack(fill='both', expand=True)

        # Праворуч: forest plot
        SectionHeader(right, "  Forest Plot (ДІ)", accent=True).pack(
            fill='x', padx=6, pady=(6, 2))
        self.pp_forest = PlotPanel(right, toolbar=False)
        self.pp_forest.pack(fill='both', expand=True, padx=4)

        self.meta_var = tk.StringVar()
        tk.Label(self, textvariable=self.meta_var, bg=C['bg'],
                 fg=C['subtext'], font=('Consolas', 8)).pack(
            anchor='w', padx=16, pady=(0, 4))

    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані")
            return
        # Отримати R (з кешу або рахувати заново)
        R = self._get_R()
        if R is None:
            return

        self.app.status.info("Обчислення часткових кореляцій…")
        res = sc.partial_correlations(self.app.df, R, self.app.alpha)

        if 'error' in res:
            messagebox.showerror("Помилка", res['error'])
            self.app.status.err(res['error'])
            return

        cols = res['cols']

        # Теплова карта
        self.pp_heat.show(plots.heatmap_figure(
            res['PR'], cols, "Матриця часткових коефіцієнтів кореляції"))

        # Forest plot
        self.pp_forest.show(plots.partial_ci_figure(res['pairs']))

        # Таблиця
        for w in self.tbl_fr.winfo_children():
            w.destroy()
        if res['pairs']:
            table_cols = list(res['pairs'][0].keys())
            DataTable(self.tbl_fr, columns=table_cols,
                      data=res['pairs']).pack(fill='both', expand=True)

        N = len(self.app.df)
        p = len(res['cols'])
        self.meta_var.set(
            f"Порядок w={res['w']}   df = N-w-2 = {res['df_deg']}   "
            f"t_крит={res['t_crit']}   α={res['alpha']}   "
            f"(ДІ через перетворення Фішера, df_ДІ = N-w-3 = {N - res['w'] - 3})")
        self.app.status.ok("Часткові кореляції розраховано")

    def _get_R(self):
        if hasattr(self.app, '_pair_corr_res') and self.app._pair_corr_res:
            return self.app._pair_corr_res['R']
        # Рахуємо самостійно
        res = sc.pearson_matrix(self.app.df, self.app.alpha)
        self.app._pair_corr_res = res
        return res['R']


# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА 4 — МНОЖИННІ КОРЕЛЯЦІЇ
# ══════════════════════════════════════════════════════════════
class TabMultiple(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')
        tk.Label(ctrl, text="Множинні коефіцієнти кореляції",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 12, 'bold')).pack(side='left')
        ttk.Button(ctrl, text="▶  Розрахувати", style='Accent.TButton',
                   command=self._run).pack(side='right')
        ttk.Separator(self).pack(fill='x')

        paned = ttk.PanedWindow(self, orient='vertical')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        top_f = ttk.Frame(paned)
        bot_f = ttk.Frame(paned)
        paned.add(top_f, weight=1)
        paned.add(bot_f, weight=2)

        # Верх: таблиця
        SectionHeader(top_f, "  Зведена таблиця R_mult", accent=True).pack(
            fill='x', padx=6, pady=(4, 2))
        self.tbl_fr = ttk.Frame(top_f)
        self.tbl_fr.pack(fill='both', expand=True, padx=6, pady=4)

        # Низ: графік
        SectionHeader(bot_f, "  Стовпчаста діаграма R_mult та R²",
                      accent=True).pack(fill='x', padx=6, pady=(4, 2))
        self.pp = PlotPanel(bot_f)
        self.pp.pack(fill='both', expand=True, padx=4)

        self.meta_var = tk.StringVar()
        tk.Label(self, textvariable=self.meta_var, bg=C['bg'],
                 fg=C['subtext'], font=('Consolas', 8)).pack(
            anchor='w', padx=16, pady=(0, 4))

    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані")
            return
        R = self._get_R()
        if R is None:
            return

        self.app.status.info("Обчислення множинних кореляцій…")
        res = sc.multiple_correlations(self.app.df, R, self.app.alpha)

        if 'error' in res:
            messagebox.showerror("Помилка", res['error'])
            return

        # Таблиця
        for w in self.tbl_fr.winfo_children():
            w.destroy()
        if res['rows']:
            table_cols = list(res['rows'][0].keys())
            DataTable(self.tbl_fr, columns=table_cols,
                      data=res['rows']).pack(fill='both', expand=True)

        # Графік
        self.pp.show(plots.multiple_bar_figure(res['rows']))

        N = len(self.app.df)
        p = len(self.app.df.columns)
        self.meta_var.set(
            f"ν1 = p-1 = {res['df1']}   ν2 = N-p-1 = {res['df2']}   "
            f"F_крит={res['F_crit']}   α={res['alpha']}   "
            f"Формула: R²_k = 1 - |R| / |R_kk|")
        self.app.status.ok("Множинні кореляції розраховано")

    def _get_R(self):
        if hasattr(self.app, '_pair_corr_res') and self.app._pair_corr_res:
            return self.app._pair_corr_res['R']
        res = sc.pearson_matrix(self.app.df, self.app.alpha)
        self.app._pair_corr_res = res
        return res['R']

# ──────────────────────────────────────────────────────────────
#  ВКЛАДКА 5 — МНОЖИННА ЛІНІЙНА РЕГРЕСІЯ (пункт 3)
# ──────────────────────────────────────────────────────────────
import stats_regression as sr
import plots_regression as pr

class TabRegression(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')
        tk.Label(ctrl, text="Множинна лінійна регресія (пункт 3)", 
                 bg=C['bg'], fg=C['accent'], font=('Consolas', 12, 'bold')).pack(side='left')

        # Вибір залежної змінної
        tk.Label(ctrl, text="  Залежна змінна:", bg=C['bg'], fg=C['text']).pack(side='left', padx=(20, 5))
        self.dep_var = ttk.Combobox(ctrl, state='readonly', width=12)
        self.dep_var.pack(side='left')

        ttk.Button(ctrl, text="▶  Побудувати модель", style='Accent.TButton',
                   command=self._run).pack(side='right')

        ttk.Separator(self).pack(fill='x')

        # Результат
        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # self.diag_panel = PlotPanel(self.result_frame)

        # Діагностична панель
        SectionHeader(self.result_frame, "  Діагностична діаграма  ", accent=True).pack(
            fill='x', padx=4, pady=(8, 2))
        self.diag_panel = PlotPanel(self.result_frame)
        self.diag_panel.pack(fill='both', expand=True, padx=4, pady=4)


    def update_columns(self, df):
        """Оновити список залежних змінних з даних"""
        if df is not None:
            self.dep_var['values'] = list(df.columns)
            if len(df.columns) > 0:
                self.dep_var.current(0)  # вибрати першу колонку за замовчуванням
        else:
            self.dep_var['values'] = []

    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Завантажте дані спочатку")
            return
        dep = self.dep_var.get()
        if not dep:
            messagebox.showwarning("Увага", "Оберіть залежну змінну")
            return

        res = sr.multiple_linear_regression(self.app.df, dep, self.app.alpha)

        # Очисти старе вмісто
        for w in self.result_frame.winfo_children():
            w.destroy()
        
        # Таблиця коефіцієнтів
        DataTable(self.result_frame, columns=list(res['coef_table'][0].keys()),
                data=res['coef_table']).pack(fill='both', expand=True, pady=4)

        # Метрики моделі
        info = f"R² = {res['R2']}  |  R²_adj = {res['R2_adj']}  |  F = {res['F_stat']} (критичне {res['F_crit']}) → {'ЗНАЧУЩА' if res['model_sig'] else 'НЕЗНАЧУЩА'}"
        tk.Label(self.result_frame, text=info, bg=C['panel'], fg=C['green'] if res['model_sig'] else C['accent2'],
                font=('Consolas', 10, 'bold')).pack(pady=4)

        # Толерантні межи для залишкової дисперсії
        tk.Label(self.result_frame, text=f"Довірчий інтервал для σ²: [{res['sigma2_ci_lo']}; {res['sigma2_ci_hi']}]",
                bg=C['panel'], fg=C['yellow']).pack()

        # Діагностична діаграма — нова панель кожний раз
        SectionHeader(self.result_frame, "  Діагностична діаграма  ", accent=True).pack(
            fill='x', padx=4, pady=(8, 2))
        diag_panel = PlotPanel(self.result_frame)
        diag_panel.pack(fill='both', expand=True, padx=4, pady=4)
        diag_panel.show(pr.regression_diagnostic_figure(res))

        self.app.status.ok(f"Регресія для {dep} побудована (R² = {res['R2']})")


# ──────────────────────────────────────────────────────────────
#  ВКЛАДКА 6 — ВІЗУАЛІЗАЦІЯ (пункт 4)
# ──────────────────────────────────────────────────────────────
class TabVisualization(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')
        tk.Label(ctrl, text="Візуалізація багатовимірних даних (пункт 4)", 
                bg=C['bg'], fg=C['accent'], font=('Consolas', 12, 'bold')).pack(side='left')
        ttk.Button(ctrl, text="▶  Показати всі графіки", style='Accent.TButton',
                command=self._run).pack(side='right')

        ttk.Separator(self).pack(fill='x')

        # Notebook для трьох графіків
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=8, pady=8)
        
        self.tab_pc = ttk.Frame(nb)
        self.tab_heat = ttk.Frame(nb)
        self.tab_bubble = ttk.Frame(nb)
        
        nb.add(self.tab_pc, text="  Паралельні координати  ")
        nb.add(self.tab_heat, text="  Теплова карта  ")
        nb.add(self.tab_bubble, text="  Бульбашкова діаграма  ")
        
        self.pc_panel = PlotPanel(self.tab_pc)
        self.pc_panel.pack(fill='both', expand=True)
        
        self.heat_panel = PlotPanel(self.tab_heat)
        self.heat_panel.pack(fill='both', expand=True)
        
        self.bubble_panel = PlotPanel(self.tab_bubble)
        self.bubble_panel.pack(fill='both', expand=True)

    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Завантажте дані")
            return
        self.pc_panel.show(pr.parallel_coordinates_figure(self.app.df))
        self.heat_panel.show(pr.heatmap_data_figure(self.app.df))  # ← НОВА ЛІНІЯ
        self.bubble_panel.show(pr.bubble_chart_figure(self.app.df))
        self.app.status.ok("Візуалізація побудована")


# ══════════════════════════════════════════════════════════════
#  ГОЛОВНЕ ВІКНО
# ══════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторна робота №5  |  Аналіз даних")
        self.geometry("1280x800")
        self.minsize(900, 600)

        # Стан
        self.df    = None
        self.alpha = 0.05
        self._pair_corr_res = None

        apply_theme(self)
        self._build_ui()

    def _build_ui(self):
        # ── Заголовок ──────────────────────────────────────────
        header = tk.Frame(self, bg=C['header_bg'], height=46)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(header,
                 text="  📊  Лаб. №5  —  Аналіз багатовимірних об'єктів спостережень",
                 bg=C['header_bg'], fg=C['text'],
                 font=('Consolas', 11, 'bold')).pack(side='left', padx=8)

        tk.Label(header,
                 text="Предмет: Аналіз даних",
                 bg=C['header_bg'], fg=C['subtext'],
                 font=('Consolas', 8)).pack(side='right', padx=12)

        # ── Notebook вкладок ────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=0, pady=0)

        tab_defs = [
            ("  📂  Дані  ",          TabLoad),
            ("  📈  Первинний аналіз  ", TabPrimary),
            ("  🔗  Парні кореляції  ", TabPairCorr),
            ("  ⚙️  Часткові кореляції  ", TabPartial),
            ("  🔢  Множинні кореляції  ", TabMultiple),
            ("  📐  Множинна регресія  ", TabRegression),      # ← новий
            ("  📊  Візуалізація  ", TabVisualization),       # ← новий
            ("  📖  Теорія  ",         TabTheory),
        ]
        for label, Cls in tab_defs:
            tab = Cls(self.notebook, self)
            self.notebook.add(tab, text=label)


        # ── Статус-бар ──────────────────────────────────────────
        self.status = StatusBar(self)
        self.status.pack(fill='x', side='bottom')
        self.status.info("Оберіть файл даних на вкладці «Дані» та натисніть «Завантажити»")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
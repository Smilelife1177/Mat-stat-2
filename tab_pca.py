"""
tab_pca.py — Вкладка «МГК» (пункт 5, Лаб. 5)

Вкладки всередині:
  0. Таблиця МГК      — факторні навантаження (формат Табл. 4.5)
  1. Scree Plot        — власні числа / % дисперсії
  2. До перетворення   — кор. поле, 2D гіст., центр, кор. матриця
  3. Після МГК         — те саме в системі ГК
  4. Зворотній перехід — те саме у відновленому оригінальному просторі
"""

import tkinter as tk
from tkinter import ttk, messagebox

import stats_pca as spca
import plots_pca as ppca
from gui_widgets import C, SectionHeader, DataTable, PlotPanel


class TabPCA(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app  = app
        self._res = None
        self._build()

    # ──────────────────────────────────────────────────────────
    def _build(self):
        # ── Заголовок + кнопка ────────────────────────────────
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')

        tk.Label(ctrl,
                 text="МГК — Перехід до незалежної системи координат (2D)",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 12, 'bold')).pack(side='left')

        ttk.Button(ctrl, text="▶  Виконати МГК",
                   style='Accent.TButton',
                   command=self._run).pack(side='right')

        ttk.Separator(self).pack(fill='x')

        # ── Основний paned: ліворуч вибір ознак, праворуч результати ─
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        left  = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left,  weight=1)
        paned.add(right, weight=5)

        # ── ЛІВА: вибір ознак ────────────────────────────────
        SectionHeader(left, "  Ознаки для МГК", accent=True).pack(
            fill='x', padx=6, pady=(6, 2))

        tk.Label(left,
                 text="Ctrl+Click — множинний вибір\n(мін. 2 ознаки)",
                 bg=C['bg'], fg=C['subtext'],
                 font=('Consolas', 8), justify='left').pack(
            anchor='w', padx=8, pady=(0, 4))

        list_frame = tk.Frame(left, bg=C['bg'])
        list_frame.pack(fill='both', expand=True, padx=6, pady=2)
        vsb_l = ttk.Scrollbar(list_frame, orient='vertical')

        self.col_listbox = tk.Listbox(
            list_frame,
            selectmode='multiple',
            bg=C['panel2'], fg=C['text'],
            selectbackground=C['accent'],
            selectforeground=C['white'],
            font=('Consolas', 10),
            relief='flat', bd=0,
            activestyle='none',
            highlightthickness=0,
            yscrollcommand=vsb_l.set,
        )
        vsb_l.configure(command=self.col_listbox.yview)
        self.col_listbox.pack(side='left', fill='both', expand=True)
        vsb_l.pack(side='right', fill='y')

        btn_row = tk.Frame(left, bg=C['bg'])
        btn_row.pack(fill='x', padx=6, pady=6)
        ttk.Button(btn_row, text="Всі",
                   command=self._select_all).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Жодного",
                   command=self._deselect_all).pack(side='left', padx=2)

        # Мета-інфо під списком
        self.meta_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.meta_var,
                 bg=C['bg'], fg=C['subtext'],
                 font=('Consolas', 7), wraplength=160,
                 justify='left').pack(anchor='w', padx=8, pady=(2, 0))

        # ── ПРАВА: notebook з результатами ───────────────────
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill='both', expand=True)

        self._tab_table   = ttk.Frame(self.nb)
        self._tab_scree   = ttk.Frame(self.nb)
        self._tab_before  = ttk.Frame(self.nb)
        self._tab_after   = ttk.Frame(self.nb)
        self._tab_inverse = ttk.Frame(self.nb)

        self.nb.add(self._tab_table,   text="  Таблиця МГК (4.5)  ")
        self.nb.add(self._tab_scree,   text="  Scree Plot  ")
        self.nb.add(self._tab_before,  text="  До перетворення  ")
        self.nb.add(self._tab_after,   text="  Після МГК  ")
        self.nb.add(self._tab_inverse, text="  Зворотній перехід  ")

        # Фрейм для таблиці
        self._tbl_frame = ttk.Frame(self._tab_table)
        self._tbl_frame.pack(fill='both', expand=True, padx=6, pady=6)

        # PlotPanel-и
        self.pp_scree   = PlotPanel(self._tab_scree)
        self.pp_scree.pack(fill='both', expand=True)

        self.pp_before  = PlotPanel(self._tab_before)
        self.pp_before.pack(fill='both', expand=True)

        self.pp_after   = PlotPanel(self._tab_after)
        self.pp_after.pack(fill='both', expand=True)

        self.pp_inverse = PlotPanel(self._tab_inverse)
        self.pp_inverse.pack(fill='both', expand=True)

    # ──────────────────────────────────────────────────────────
    def update_columns(self, df):
        """Викликається при завантаженні нових даних."""
        self.col_listbox.delete(0, 'end')
        if df is not None:
            for col in df.columns:
                self.col_listbox.insert('end', col)
            self.col_listbox.select_set(0, 'end')   # вибрати всі за замовч.

    def _select_all(self):
        self.col_listbox.select_set(0, 'end')

    def _deselect_all(self):
        self.col_listbox.select_clear(0, 'end')

    # ──────────────────────────────────────────────────────────
    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані")
            return

        sel_idx = list(self.col_listbox.curselection())
        if len(sel_idx) < 2:
            messagebox.showwarning("Увага", "Оберіть щонайменше 2 ознаки для МГК")
            return

        selected_cols = [self.col_listbox.get(i) for i in sel_idx]
        self.app.status.info(f"МГК: обробка {len(selected_cols)} ознак…")

        try:
            res = spca.pca_analysis(self.app.df, selected_cols, self.app.alpha)
        except Exception as exc:
            messagebox.showerror("Помилка МГК", str(exc))
            self.app.status.err(str(exc))
            return

        self._res = res
        self._show_results(res)

        pc1_pct = res['var_pct'][0]
        pc2_pct = res['cum_pct'][min(1, res['p'] - 1)]
        self.meta_var.set(
            f"n={res['n']}  p={res['p']}\n"
            f"PC1={pc1_pct:.1f}%\n"
            f"PC1+PC2={pc2_pct:.1f}%")
        self.app.status.ok(
            f"МГК: {res['p']} ознак → "
            f"PC1={pc1_pct:.1f}%, PC1+PC2={pc2_pct:.1f}%")

    # ──────────────────────────────────────────────────────────
    def _show_results(self, res):
        p         = res['p']
        cols      = res['cols']
        pc_names  = res['pc_names']

        # ── Таблиця МГК ──────────────────────────────────────
        for w in self._tbl_frame.winfo_children():
            w.destroy()
        tbl_cols = ['Ознака'] + pc_names
        DataTable(self._tbl_frame,
                  columns=tbl_cols,
                  data=res['table_rows']).pack(fill='both', expand=True)

        # ── Scree plot ────────────────────────────────────────
        self.pp_scree.show(
            ppca.pca_variance_figure(
                res['eigenvalues'], res['var_pct'], res['cum_pct']))

        # ── Вибираємо 2 виміри для 2D-панелей ────────────────
        # "До перетворення" — оригінальні дані (стандартизовані)
        self.pp_before.show(
            ppca.pca_panel_figure(
                data=res['X_standardized'],
                col_names=cols,
                center=res['center_orig'],          # центр оригінал. простору
                corr_matrix=res['R_orig'],
                title="До перетворення — оригінальний стандартизований простір",
            ))

        # "Після МГК" — координати в системі ГК
        self.pp_after.show(
            ppca.pca_panel_figure(
                data=res['Z'],
                col_names=pc_names,
                center=res['center_pca'],
                corr_matrix=res['R_pca'],
                title="Після МГК — система головних компонент (незалежні осі)",
            ))

        # "Зворотній перехід" — відновлені оригінальні дані
        self.pp_inverse.show(
            ppca.pca_panel_figure(
                data=res['X_reconstructed'],
                col_names=cols,
                center=res['center_rec'],
                corr_matrix=res['R_rec'],
                title="Зворотній перехід — відновлені дані в початковому просторі",
            ))

        self.nb.select(0)
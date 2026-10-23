"""
tab_pca.py — Вкладка «МГК (PCA)» для виконання лабораторної роботи.

Вимоги лабораторної роботи:
  1. На початкових даних матрицю діаграм розкиду.
  2. Провести МГК, результати представити у вигляді таблиці: власні значення, власні вектори, частку загальної дисперсії.
  3. Представити матрицю діаграм розкиду для ознак, які складають більше 95% частки від загальної дисперсії.
  4. В трьохвимірному просторі реалізацію (3D-візуалізація спостережень).
  5. Зробити зворотне перетворення (відновлення початкових даних з відібраних ГК).
  6. Матрицю діаграм розкиду для відновлених даних.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

import stats_pca as spca
import plots_pca as ppca
from gui_widgets import C, SectionHeader, DataTable, PlotPanel, MetricCard


class TabPCA(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._res = None
        self._fig_3d = None
        self._build()

    # ──────────────────────────────────────────────────────────
    def _build(self):
        # ── Заголовок + кнопки ────────────────────────────────
        ctrl = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        ctrl.pack(fill='x')

        title_box = tk.Frame(ctrl, bg=C['bg'])
        title_box.pack(side='left')
        tk.Label(title_box,
                 text="🧬 МГК — Метод Головних Компонент (PCA)",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 13, 'bold')).pack(anchor='w')
        tk.Label(title_box,
                 text="Зниження розмірності, аналіз власних чисел, 3D-проекція та зворотне перетворення",
                 bg=C['bg'], fg=C['subtext'],
                 font=('Consolas', 9)).pack(anchor='w')

        ttk.Button(ctrl, text="▶  Виконати МГК",
                   style='Accent.TButton',
                   command=self._run).pack(side='right', padx=4)

        ttk.Separator(self).pack(fill='x')

        # ── Основний paned: ліворуч налаштування, праворуч результати ─
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=5)

        # ══════════════════════════════════════════════════════
        # ЛІВА ПАНЕЛЬ: вибір ознак та параметрів
        # ══════════════════════════════════════════════════════
        SectionHeader(left, "  Ознаки для МГК", accent=True).pack(
            fill='x', padx=6, pady=(4, 2))

        tk.Label(left,
                 text="Ctrl+Click — множинний вибір\n(для 3D-простору мін. 3 ознаки)",
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
            height=6,
        )
        vsb_l.configure(command=self.col_listbox.yview)
        self.col_listbox.pack(side='left', fill='both', expand=True)
        vsb_l.pack(side='right', fill='y')

        btn_row = tk.Frame(left, bg=C['bg'])
        btn_row.pack(fill='x', padx=6, pady=4)
        ttk.Button(btn_row, text="Всі",
                   command=self._select_all).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Жодного",
                   command=self._deselect_all).pack(side='left', padx=2)

        # ── Параметри відбору компонент ───────────────────────
        SectionHeader(left, "  Параметри відбору").pack(
            fill='x', padx=6, pady=(8, 2))

        param_box = tk.Frame(left, bg=C['panel'], padx=8, pady=8)
        param_box.pack(fill='x', padx=6, pady=4)

        tk.Label(param_box, text="Поріг дисперсії (%):",
                 bg=C['panel'], fg=C['text'],
                 font=('Consolas', 9)).pack(anchor='w')
        self.thresh_var = tk.StringVar(value="95.0")
        thresh_cb = ttk.Combobox(param_box, textvariable=self.thresh_var,
                                 values=['80.0', '90.0', '95.0', '98.0', '99.0'],
                                 width=12, state='readonly')
        thresh_cb.pack(fill='x', pady=(2, 6))

        tk.Label(param_box, text="Кількість ГК (k):",
                 bg=C['panel'], fg=C['text'],
                 font=('Consolas', 9)).pack(anchor='w')
        self.k_var = tk.StringVar(value="Авто (> порогу)")
        self.k_cb = ttk.Combobox(param_box, textvariable=self.k_var,
                                 values=['Авто (> порогу)', '1', '2', '3', '4'],
                                 width=12, state='readonly')
        self.k_cb.pack(fill='x', pady=(2, 2))

        # ── Сводна статистика МГК ──────────────────────────────
        SectionHeader(left, "  Підсумкова статистика").pack(
            fill='x', padx=6, pady=(8, 2))

        self.info_box = tk.Frame(left, bg=C['panel'], padx=8, pady=8)
        self.info_box.pack(fill='x', padx=6, pady=4)

        self.lbl_dims = tk.Label(self.info_box, text="Розмірність: —",
                                 bg=C['panel'], fg=C['text'], font=('Consolas', 9), anchor='w')
        self.lbl_dims.pack(fill='x', pady=1)

        self.lbl_k = tk.Label(self.info_box, text="Відібрано ГК: —",
                              bg=C['panel'], fg=C['accent'], font=('Consolas', 9, 'bold'), anchor='w')
        self.lbl_k.pack(fill='x', pady=1)

        self.lbl_var = tk.Label(self.info_box, text="Пояснена дисперсія: —",
                                bg=C['panel'], fg=C['green'], font=('Consolas', 9), anchor='w')
        self.lbl_var.pack(fill='x', pady=1)

        self.lbl_loss = tk.Label(self.info_box, text="Втрата дисперсії: —",
                                 bg=C['panel'], fg=C['yellow'], font=('Consolas', 9), anchor='w')
        self.lbl_loss.pack(fill='x', pady=1)

        self.lbl_mse = tk.Label(self.info_box, text="Сер. похибка MSE: —",
                                bg=C['panel'], fg=C['text'], font=('Consolas', 9), anchor='w')
        self.lbl_mse.pack(fill='x', pady=1)

        # ══════════════════════════════════════════════════════
        # ПРАВА ПАНЕЛЬ: Вкладки згідно з вимогами лаби
        # ══════════════════════════════════════════════════════
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill='both', expand=True)

        self._tab_orig_scatter = ttk.Frame(self.nb)
        self._tab_table_scree  = ttk.Frame(self.nb)
        self._tab_pca_scatter  = ttk.Frame(self.nb)
        self._tab_3d           = ttk.Frame(self.nb)
        self._tab_inverse      = ttk.Frame(self.nb)
        self._tab_rec_scatter  = ttk.Frame(self.nb)
        self._tab_2d           = ttk.Frame(self.nb)

        self.nb.add(self._tab_orig_scatter, text="  📊 1. Початкові дані (Scatter)  ")
        self.nb.add(self._tab_table_scree,  text="  📋 2. Таблиця МГК & Scree  ")
        self.nb.add(self._tab_pca_scatter,  text="  🎯 3. ГК >95% (Scatter)  ")
        self.nb.add(self._tab_3d,           text="  🌐 4. 3D-простір (3 ГК)  ")
        self.nb.add(self._tab_inverse,      text="  🔄 5. Зворотне перетворення  ")
        self.nb.add(self._tab_rec_scatter,  text="  ✨ 6. Відновлені дані (Scatter)  ")
        self.nb.add(self._tab_2d,           text="  🔍 7. 2D-зрізи  ")

        # ── Вкладка 1: Scatter матриця початкових даних ───────
        self.pp_orig_scatter = PlotPanel(self._tab_orig_scatter)
        self.pp_orig_scatter.pack(fill='both', expand=True)

        # ── Вкладка 2: Таблиця МГК + Scree plot ───────────────
        tab2_paned = ttk.PanedWindow(self._tab_table_scree, orient='vertical')
        tab2_paned.pack(fill='both', expand=True, padx=4, pady=4)

        top_tbl_frame = ttk.Frame(tab2_paned)
        bot_scree_frame = ttk.Frame(tab2_paned)
        tab2_paned.add(top_tbl_frame, weight=3)
        tab2_paned.add(bot_scree_frame, weight=4)

        self.cards_frame = tk.Frame(top_tbl_frame, bg=C['bg'])
        self.cards_frame.pack(fill='x', padx=4, pady=(2, 6))

        self._tbl_frame = ttk.Frame(top_tbl_frame)
        self._tbl_frame.pack(fill='both', expand=True, padx=4, pady=2)

        self.pp_scree = PlotPanel(bot_scree_frame)
        self.pp_scree.pack(fill='both', expand=True)

        # ── Вкладка 3: Scatter матриця ГК (>95%) ───────────────
        self.pp_pca_scatter = PlotPanel(self._tab_pca_scatter)
        self.pp_pca_scatter.pack(fill='both', expand=True)

        # ── Вкладка 4: 3D-простір ГК ──────────────────────────
        ctrl_3d = tk.Frame(self._tab_3d, bg=C['bg'], pady=4, padx=8)
        ctrl_3d.pack(fill='x')
        tk.Label(ctrl_3d, text="Камера 3D:", bg=C['bg'], fg=C['text'],
                 font=('Consolas', 9)).pack(side='left', padx=(0, 6))

        ttk.Button(ctrl_3d, text="Ізометрія (3D)",
                   command=lambda: self._set_3d_view(24, 46)).pack(side='left', padx=2)
        ttk.Button(ctrl_3d, text="PC1 × PC2 (Зверху)",
                   command=lambda: self._set_3d_view(90, -90)).pack(side='left', padx=2)
        ttk.Button(ctrl_3d, text="PC1 × PC3 (Збоку)",
                   command=lambda: self._set_3d_view(0, -90)).pack(side='left', padx=2)
        ttk.Button(ctrl_3d, text="PC2 × PC3 (Фронтально)",
                   command=lambda: self._set_3d_view(0, 0)).pack(side='left', padx=2)

        tk.Label(ctrl_3d, text="(або затисніть ліву кнопку миші на графіку для вільного обертання)",
                 bg=C['bg'], fg=C['subtext'], font=('Consolas', 8)).pack(side='left', padx=(10, 0))

        self.pp_3d = PlotPanel(self._tab_3d)
        self.pp_3d.pack(fill='both', expand=True)

        # ── Вкладка 5: Зворотне перетворення (точність) ────────
        self.inv_cards_frame = tk.Frame(self._tab_inverse, bg=C['bg'])
        self.inv_cards_frame.pack(fill='x', padx=4, pady=(4, 6))

        inv_paned = ttk.PanedWindow(self._tab_inverse, orient='horizontal')
        inv_paned.pack(fill='both', expand=True, padx=4, pady=4)

        inv_left = ttk.Frame(inv_paned)
        inv_right = ttk.Frame(inv_paned)
        inv_paned.add(inv_left, weight=4)
        inv_paned.add(inv_right, weight=5)

        SectionHeader(inv_left, "  Метрики точності відновлення (X vs X̂)").pack(fill='x', pady=(0, 4))
        self._inv_tbl_frame = ttk.Frame(inv_left)
        self._inv_tbl_frame.pack(fill='both', expand=True)

        SectionHeader(inv_right, "  Точкові графіки відповідності (y = x)").pack(fill='x', pady=(0, 4))
        self.pp_inverse_diag = PlotPanel(inv_right)
        self.pp_inverse_diag.pack(fill='both', expand=True)

        # ── Вкладка 6: Scatter матриця відновлених даних ───────
        self.pp_rec_scatter = PlotPanel(self._tab_rec_scatter)
        self.pp_rec_scatter.pack(fill='both', expand=True)

        # ── Вкладка 7: 2D-зрізи (До / Після / Відновл.) ────────
        sub_2d_nb = ttk.Notebook(self._tab_2d)
        sub_2d_nb.pack(fill='both', expand=True)

        self._tab_2d_before = ttk.Frame(sub_2d_nb)
        self._tab_2d_after  = ttk.Frame(sub_2d_nb)
        self._tab_2d_rec    = ttk.Frame(sub_2d_nb)

        sub_2d_nb.add(self._tab_2d_before, text="  До перетворення  ")
        sub_2d_nb.add(self._tab_2d_after,  text="  Після МГК  ")
        sub_2d_nb.add(self._tab_2d_rec,    text="  Відновлені дані  ")

        self.pp_2d_before = PlotPanel(self._tab_2d_before)
        self.pp_2d_before.pack(fill='both', expand=True)

        self.pp_2d_after = PlotPanel(self._tab_2d_after)
        self.pp_2d_after.pack(fill='both', expand=True)

        self.pp_2d_rec = PlotPanel(self._tab_2d_rec)
        self.pp_2d_rec.pack(fill='both', expand=True)

    # ──────────────────────────────────────────────────────────
    def update_columns(self, df):
        """Оновлення списку колонок при завантаженні нового датасету."""
        self.col_listbox.delete(0, 'end')
        if df is not None:
            for col in df.columns:
                self.col_listbox.insert('end', col)
            self.col_listbox.select_set(0, 'end')

            # Оновлюємо вибір кількості ГК
            p = len(df.columns)
            k_vals = ['Авто (> порогу)'] + [str(i) for i in range(1, p + 1)]
            self.k_cb.config(values=k_vals)
            self.k_var.set('Авто (> порогу)')

    def _select_all(self):
        self.col_listbox.select_set(0, 'end')

    def _deselect_all(self):
        self.col_listbox.select_clear(0, 'end')

    def _set_3d_view(self, elev, azim):
        """Зміна кута огляду 3D камери."""
        if self._fig_3d is not None and hasattr(self._fig_3d, '_pca_3d_ax'):
            self._fig_3d._pca_3d_ax.view_init(elev=elev, azim=azim)
            if self.pp_3d._canvas:
                self.pp_3d._canvas.draw_idle()

    # ──────────────────────────────────────────────────────────
    def _run(self):
        if self.app.df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані у вкладці «Дані»")
            return

        sel_idx = list(self.col_listbox.curselection())
        if len(sel_idx) < 2:
            messagebox.showwarning("Увага", "Оберіть щонайменше 2 ознаки для виконання МГК")
            return

        selected_cols = [self.col_listbox.get(i) for i in sel_idx]

        try:
            thresh = float(self.thresh_var.get())
        except ValueError:
            thresh = 95.0

        k_val = self.k_var.get()
        k_comp = None
        if k_val and k_val != 'Авто (> порогу)':
            try:
                k_comp = int(k_val)
            except ValueError:
                k_comp = None

        self.app.status.info(f"МГК: обробка {len(selected_cols)} ознак (поріг {thresh}%)...")

        try:
            res = spca.pca_analysis(self.app.df, selected_cols,
                                    variance_threshold=thresh,
                                    k_components=k_comp,
                                    alpha=self.app.alpha)
        except Exception as exc:
            messagebox.showerror("Помилка МГК", str(exc))
            self.app.status.err(str(exc))
            return

        self._res = res
        self._show_results(res)

        # Оновлення підсумкової статистики ліворуч
        self.lbl_dims.config(text=f"Розмірність: n={res['n']}, p={res['p']}")
        self.lbl_k.config(text=f"Відібрано ГК: k={res['k_retained']} з {res['p']}")
        self.lbl_var.config(text=f"Пояснена: {res['retained_var_pct']:.2f}%")
        self.lbl_loss.config(text=f"Втрата: {res['lost_var_pct']:.2f}%")
        self.lbl_mse.config(text=f"Сер. MSE: {res['mean_mse']:.6f}")

        self.app.status.ok(
            f"МГК виконано успішно: {res['p']} ознак → відібрано {res['k_retained']} ГК "
            f"({res['retained_var_pct']:.2f}% дисперсії). Сер. MSE={res['mean_mse']:.5f}")

    # ──────────────────────────────────────────────────────────
    def _show_results(self, res):
        p                 = res['p']
        cols              = res['cols']
        pc_names          = res['pc_names']
        k_ret             = res['k_retained']
        retained_var      = res['retained_var_pct']
        lost_var          = res['lost_var_pct']

        # ── 1. Scatter-матриця початкових даних ────────────────
        fig_orig = ppca.pca_scatter_matrix_figure(
            res['df_orig'],
            title="1. Матриця діаграм розкиду початкових даних",
            subtitle=f"Розмірність: {res['n']} спостережень × {p} ознак  |  Діагональ: гістограми, Верхній кут: r Пірсона",
            color=C['accent']
        )
        self.pp_orig_scatter.show(fig_orig)

        # ── 2. Таблиця МГК та Scree plot ───────────────────────
        # Картки показників
        for w in self.cards_frame.winfo_children():
            w.destroy()
        MetricCard(self.cards_frame, "Загальна дисперсія", f"{float(res['eigenvalues'].sum()):.3f}").pack(side='left', padx=4)
        MetricCard(self.cards_frame, f"Відібрано ГК (>{res['variance_threshold']}%)", f"{k_ret} ГК ({retained_var:.1f}%)", color=C['green']).pack(side='left', padx=4)
        MetricCard(self.cards_frame, "Втрачена дисперсія", f"{lost_var:.2f}%", color=C['yellow']).pack(side='left', padx=4)
        MetricCard(self.cards_frame, "Критерій Кайзера (λ≥1)", f"{res['kaiser_count']} ГК").pack(side='left', padx=4)

        # Таблиця
        for w in self._tbl_frame.winfo_children():
            w.destroy()
        tbl_cols = ['Ознака'] + pc_names
        DataTable(self._tbl_frame,
                  columns=tbl_cols,
                  data=res['table_rows']).pack(fill='both', expand=True)

        # Scree plot
        fig_scree = ppca.pca_variance_figure(
            res['eigenvalues'], res['var_pct'], res['cum_pct'],
            threshold=res['variance_threshold'],
            k_retained=k_ret
        )
        self.pp_scree.show(fig_scree)

        # ── 3. Scatter-матриця ГК (>95% дисперсії) ─────────────
        fig_pca_scatter = ppca.pca_scatter_matrix_figure(
            res['df_pca_retained'],
            title=f"3. Матриця діаграм розкиду відібраних ГК (>{res['variance_threshold']}%)",
            subtitle=f"Відібрано {k_ret} компонент (Сумарна дисперсія: {retained_var:.2f}%)  |  Всі недіагональні кореляції r = 0.000",
            color=C['green']
        )
        self.pp_pca_scatter.show(fig_pca_scatter)

        # ── 4. 3D-простір головних компонент ───────────────────
        fig_3d = ppca.pca_3d_figure(
            res['Z'], res['var_pct'], res['pc_names'],
            title="4. 3D-реалізація вибірки в системі координат головних компонент"
        )
        self._fig_3d = fig_3d
        self.pp_3d.show(fig_3d)

        # ── 5. Зворотне перетворення (метрики та графіки) ─────
        for w in self.inv_cards_frame.winfo_children():
            w.destroy()
        MetricCard(self.inv_cards_frame, "Зниження вимірності", f"{p} ознак → {k_ret} ГК").pack(side='left', padx=4)
        MetricCard(self.inv_cards_frame, "Збережена дисперсія", f"{retained_var:.2f}%", color=C['green']).pack(side='left', padx=4)
        MetricCard(self.inv_cards_frame, "Втрата дисперсії", f"{lost_var:.2f}%", color=C['yellow']).pack(side='left', padx=4)
        MetricCard(self.inv_cards_frame, "Середній MSE відновлення", f"{res['mean_mse']:.6f}").pack(side='left', padx=4)

        for w in self._inv_tbl_frame.winfo_children():
            w.destroy()
        inv_cols = ['Ознака', 'Сер. ориг.', 'Сер. відновл.', 'СКВ ориг.', 'СКВ відновл.',
                    'MSE', 'RMSE', 'MAE', 'R²', 'r(X, X̂)']
        DataTable(self._inv_tbl_frame,
                  columns=inv_cols,
                  data=res['reconstruction_rows']).pack(fill='both', expand=True)

        fig_diag = ppca.pca_reconstruction_figure(
            res['X_orig'], res['X_reconstructed'], cols,
            k_retained=k_ret, retained_var=retained_var
        )
        self.pp_inverse_diag.show(fig_diag)

        # ── 6. Scatter-матриця відновлених даних ───────────────
        fig_rec_scatter = ppca.pca_scatter_matrix_figure(
            res['df_reconstructed'],
            title="6. Матриця діаграм розкиду відновлених даних",
            subtitle=f"Відновлено з k={k_ret} ГК (пояснюють {retained_var:.2f}% дисперсії)  |  Всі взаємозв'язки та розподіли збережено",
            color=C['yellow']
        )
        self.pp_rec_scatter.show(fig_rec_scatter)

        # ── 7. 2D-зрізи (До / Після / Відновл.) ─────────────────
        self.pp_2d_before.show(
            ppca.pca_panel_figure(
                data=res['X_standardized'],
                col_names=cols,
                center=res['center_orig'],
                corr_matrix=res['R_orig'],
                title="До перетворення — оригінальний стандартизований простір",
            ))

        self.pp_2d_after.show(
            ppca.pca_panel_figure(
                data=res['Z'],
                col_names=pc_names,
                center=res['center_pca'],
                corr_matrix=res['R_pca'],
                title="Після МГК — система головних компонент (незалежні осі)",
            ))

        self.pp_2d_rec.show(
            ppca.pca_panel_figure(
                data=res['X_reconstructed'],
                col_names=cols,
                center=res['center_rec'],
                corr_matrix=res['R_rec'],
                title="Зворотній перехід — відновлені дані в початковому просторі",
            ))

        self.nb.select(0)

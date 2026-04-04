"""
gui_widgets.py — Перевикористовувані tkinter-компоненти
Стиль: темна академічна тема
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ── Палітра кольорів ──────────────────────────────────────────
C = {
    'bg':       '#0f1117',
    'panel':    '#1a1d27',
    'panel2':   '#20243a',
    'border':   '#2c3050',
    'accent':   '#4f8ef7',
    'accent2':  '#e05c7a',
    'green':    '#4ecb88',
    'yellow':   '#f5c542',
    'text':     '#d6dae8',
    'subtext':  '#8891aa',
    'white':    '#ffffff',
    'header_bg':'#141724',
}


def apply_theme(root):
    """Налаштовує глобальну тему ttk і tk."""
    root.configure(bg=C['bg'])
    style = ttk.Style(root)
    style.theme_use('clam')

    style.configure('.',
        background=C['bg'], foreground=C['text'],
        fieldbackground=C['panel'], troughcolor=C['panel'],
        bordercolor=C['border'], darkcolor=C['panel'],
        lightcolor=C['panel2'], font=('Consolas', 9))

    style.configure('TFrame',   background=C['bg'])
    style.configure('Panel.TFrame', background=C['panel'],
                    relief='flat', borderwidth=0)

    style.configure('TLabel',
        background=C['bg'], foreground=C['text'], font=('Consolas', 9))
    style.configure('Header.TLabel',
        background=C['bg'], foreground=C['accent'],
        font=('Consolas', 13, 'bold'))
    style.configure('Sub.TLabel',
        background=C['bg'], foreground=C['subtext'], font=('Consolas', 8))
    style.configure('Val.TLabel',
        background=C['panel'], foreground=C['yellow'],
        font=('Consolas', 10, 'bold'))

    style.configure('TButton',
        background=C['panel2'], foreground=C['text'],
        relief='flat', borderwidth=1,
        font=('Consolas', 9, 'bold'), padding=(8, 4))
    style.map('TButton',
        background=[('active', C['accent']), ('pressed', '#3a70d0')],
        foreground=[('active', C['white'])])

    style.configure('Accent.TButton',
        background=C['accent'], foreground=C['white'],
        relief='flat', font=('Consolas', 9, 'bold'), padding=(10, 5))
    style.map('Accent.TButton',
        background=[('active', '#3a70d0'), ('pressed', '#2a50a0')])

    style.configure('Danger.TButton',
        background=C['accent2'], foreground=C['white'],
        relief='flat', font=('Consolas', 9, 'bold'), padding=(8, 4))

    style.configure('TNotebook',
        background=C['bg'], borderwidth=0, tabmargins=0)
    style.configure('TNotebook.Tab',
        background=C['panel'], foreground=C['subtext'],
        font=('Consolas', 9, 'bold'), padding=(14, 6),
        borderwidth=0)
    style.map('TNotebook.Tab',
        background=[('selected', C['accent'])],
        foreground=[('selected', C['white'])])

    style.configure('Treeview',
        background=C['panel'], foreground=C['text'],
        rowheight=22, fieldbackground=C['panel'],
        borderwidth=0, font=('Consolas', 8))
    style.configure('Treeview.Heading',
        background=C['header_bg'], foreground=C['accent'],
        font=('Consolas', 8, 'bold'), relief='flat')
    style.map('Treeview',
        background=[('selected', C['accent'])],
        foreground=[('selected', C['white'])])

    style.configure('Horizontal.TScrollbar',
        background=C['panel2'], troughcolor=C['panel'],
        arrowcolor=C['subtext'], relief='flat')
    style.configure('Vertical.TScrollbar',
        background=C['panel2'], troughcolor=C['panel'],
        arrowcolor=C['subtext'], relief='flat')

    style.configure('TSeparator', background=C['border'])
    style.configure('TEntry',
        fieldbackground=C['panel2'], foreground=C['text'],
        insertcolor=C['accent'], borderwidth=1, relief='flat')
    style.configure('TCombobox',
        fieldbackground=C['panel2'], foreground=C['text'],
        arrowcolor=C['accent'])
    style.configure('TSpinbox',
        fieldbackground=C['panel2'], foreground=C['text'],
        arrowcolor=C['accent'])


# ──────────────────────────────────────────────────────────────
# КАРТКА-МЕТРИКА (один показник)
# ──────────────────────────────────────────────────────────────
class MetricCard(ttk.Frame):
    def __init__(self, parent, label, value, color=None, **kw):
        super().__init__(parent, style='Panel.TFrame', **kw)
        self.configure(padding=(12, 8))
        color = color or C['accent']
        tk.Label(self, text=label, bg=C['panel'], fg=C['subtext'],
                 font=('Consolas', 8)).pack(anchor='w')
        tk.Label(self, text=str(value), bg=C['panel'], fg=color,
                 font=('Consolas', 13, 'bold')).pack(anchor='w')


# ──────────────────────────────────────────────────────────────
# ТАБЛИЦЯ (Treeview з прокруткою)
# ──────────────────────────────────────────────────────────────
class DataTable(ttk.Frame):
    """
    Параметри:
      parent   – батьківський віджет
      columns  – list[str] назв колонок
      data     – list[dict] або list[list] рядків
      stripe   – чергування рядків
    """

    def __init__(self, parent, columns: list, data: list,
                 stripe=True, **kw):
        super().__init__(parent, **kw)
        self._stripe = stripe

        # Treeview + scrollbars
        vsb = ttk.Scrollbar(self, orient='vertical')
        hsb = ttk.Scrollbar(self, orient='horizontal')
        self.tree = ttk.Treeview(
            self,
            columns=columns,
            show='headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            selectmode='browse',
        )
        vsb.configure(command=self.tree.yview)
        hsb.configure(command=self.tree.xview)

        # Заголовки
        for col in columns:
            w = max(80, len(str(col)) * 9)
            self.tree.heading(col, text=col, anchor='center')
            self.tree.column(col, width=w, anchor='center', minwidth=50)

        self.tree.tag_configure('odd',  background=C['panel'])
        self.tree.tag_configure('even', background=C['panel2'])
        self.tree.tag_configure('sig',  foreground=C['green'])
        self.tree.tag_configure('nsig', foreground=C['accent2'])

        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.load(data)

    def load(self, data: list):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for idx, row in enumerate(data):
            if isinstance(row, dict):
                values = list(row.values())
            else:
                values = list(row)

            tags = ['odd' if idx % 2 == 0 else 'even']
            # Підсвічування значущості
            for v in values:
                if '✔' in str(v):
                    tags.append('sig'); break
                if '✘' in str(v):
                    tags.append('nsig'); break

            self.tree.insert('', 'end', values=values, tags=tags)


# ──────────────────────────────────────────────────────────────
# ПАНЕЛЬ ГРАФІКА (matplotlib у tkinter)
# ──────────────────────────────────────────────────────────────
class PlotPanel(ttk.Frame):
    """Вбудовує matplotlib Figure у tkinter-фрейм."""

    def __init__(self, parent, toolbar=True, **kw):
        super().__init__(parent, **kw)
        self._toolbar = toolbar
        self._canvas  = None
        self._bar     = None

    def show(self, fig: Figure):
        """Відображає новий графік (видаляє попередній)."""
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        if self._bar:
            self._bar.destroy()

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        widget = self._canvas.get_tk_widget()
        widget.configure(bg=C['bg'])
        widget.pack(fill='both', expand=True)

        if self._toolbar:
            # Окремий фрейм під тулбар
            bar_frame = tk.Frame(self, bg=C['panel'], pady=2)
            bar_frame.pack(fill='x', side='bottom')
            self._bar = NavigationToolbar2Tk(self._canvas, bar_frame)
            self._bar.config(bg=C['panel'])
            for child in self._bar.winfo_children():
                try:
                    child.config(bg=C['panel'], fg=C['text'],
                                 activebackground=C['panel2'])
                except Exception:
                    pass

        self._canvas.draw()

    def clear(self):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None


# ──────────────────────────────────────────────────────────────
# СЕКЦІЯ З ЗАГОЛОВКОМ
# ──────────────────────────────────────────────────────────────
class SectionHeader(tk.Frame):
    def __init__(self, parent, text, accent=False, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        color = C['accent'] if accent else C['text']
        tk.Label(self, text=text, bg=C['bg'], fg=color,
                 font=('Consolas', 11, 'bold')).pack(side='left')
        sep = tk.Frame(self, bg=C['border'], height=1)
        sep.pack(side='left', fill='x', expand=True, padx=(10, 0), pady=8)


# ──────────────────────────────────────────────────────────────
# СТАТУС-БАР
# ──────────────────────────────────────────────────────────────
class StatusBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C['header_bg'], height=24, **kw)
        self._label = tk.Label(self, text="Готово", bg=C['header_bg'],
                               fg=C['subtext'], font=('Consolas', 8),
                               anchor='w', padx=10)
        self._label.pack(side='left', fill='x', expand=True)
        self._right = tk.Label(self, text="", bg=C['header_bg'],
                               fg=C['subtext'], font=('Consolas', 8),
                               anchor='e', padx=10)
        self._right.pack(side='right')

    def set(self, msg: str, right: str = ""):
        self._label.configure(text=msg)
        if right:
            self._right.configure(text=right)

    def ok(self, msg="Готово"):
        self._label.configure(text=f"✔  {msg}", fg=C['green'])

    def err(self, msg="Помилка"):
        self._label.configure(text=f"✘  {msg}", fg=C['accent2'])

    def info(self, msg):
        self._label.configure(text=f"ℹ  {msg}", fg=C['subtext'])
"""
tab_theory.py — Вкладка «Теорія» для захисту лабораторної роботи
Містить всі типові питання викладача з поясненнями та формулами
"""

import tkinter as tk
from tkinter import ttk
from gui_widgets import C

# ──────────────────────────────────────────────────────────────
# КОНТЕНТ: питання → відповідь (текст + формула якщо є)
# ──────────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": "01",
        "short": "Матриця кореляцій",
        "question": "Що таке матриця кореляцій?",
        "answer": (
            "Квадратна матриця p×p, де елемент r_jk — коефіцієнт кореляції Пірсона між ознаками j і k.\n\n"
            "Властивості:\n"
            "  • На діагоналі завжди 1  (ознака корелює сама з собою)\n"
            "  • Симетрична: r_jk = r_kj\n"
            "  • Всі елементи від −1 до +1\n\n"
            "У програмі виводяться дві матриці:\n"
            "  1) Повна — всі r_jk\n"
            "  2) Значуща — незначущі коефіцієнти замінені на 0"
        ),
        "formula": "r_jk = cov(Xj, Xk) / (s_j · s_k)",
        "tag": "кореляція",
    },
    {
        "id": "02",
        "short": "Середнє та СКВ",
        "question": "Формули вибіркового середнього та СКВ?",
        "answer": (
            "Вибіркове середнє (оцінка математичного сподівання):\n\n"
            "  x̄_j = (1/n) · Σ x_ij\n\n"
            "Незміщене СКВ (ділимо на n−1, а не n!):\n\n"
            "  s_j = sqrt[ (1/(n−1)) · Σ(x_ij − x̄_j)² ]\n\n"
            "Чому n−1?  Бо при оцінці дисперсії ми вже витратили 1 ступінь свободи "
            "на оцінку середнього. Поділ на n дає зміщену оцінку.\n\n"
            "Стандартна похибка середнього:\n\n"
            "  SE_j = s_j / sqrt(n)\n\n"
            "Довірчий інтервал для середнього:\n\n"
            "  x̄_j ± t(α/2, n−1) · SE_j"
        ),
        "formula": "x̄ = (1/n)·Σxᵢ      s = sqrt[Σ(xᵢ−x̄)²/(n−1)]",
        "tag": "первинний аналіз",
    },
    {
        "id": "03",
        "short": "Скільки часткових коефіцієнтів?",
        "question": "Скільки часткових коефіцієнтів кореляції при 3 ознаках?",
        "answer": (
            "При p = 3 ознаках (X1, X2, X3) — рівно 3 часткові коефіцієнти:\n\n"
            "  r_12·3 — зв'язок X1 і X2, виключивши вплив X3\n"
            "  r_13·2 — зв'язок X1 і X3, виключивши вплив X2\n"
            "  r_23·1 — зв'язок X2 і X3, виключивши вплив X1\n\n"
            "Загальна формула для p ознак:\n\n"
            "  кількість пар = p·(p−1)/2\n\n"
            "При p=3:  3·2/2 = 3  ✔\n"
            "При p=4:  4·3/2 = 6\n"
            "При p=5:  5·4/2 = 10"
        ),
        "formula": "Кількість пар = p·(p−1)/2",
        "tag": "часткова кореляція",
    },
    {
        "id": "04",
        "short": "Скільки оцінок значущості?",
        "question": "Скільки оцінок значущості при α = 0.05?",
        "answer": (
            "Рівень значущості α = 0.05 — ОДИН для всіх перевірок.\n\n"
            "Але t-тест виконується окремо для кожного коефіцієнта кореляції.\n"
            "Кількість перевірок = кількість пар = p·(p−1)/2\n\n"
            "При p=3:\n"
            "  • 3 перевірки для парних кореляцій  (df = n−2)\n"
            "  • 3 перевірки для часткових кореляцій  (df = n−p)\n\n"
            "Критичне значення t_крит — однакове для всіх пар (бо df однаковий).\n"
            "Але t-статистика своя для кожної пари."
        ),
        "formula": "t = r·sqrt(n−2) / sqrt(1−r²)  ~  t(n−2)",
        "tag": "значущість",
    },
    {
        "id": "05",
        "short": "Що показує довірчий інтервал?",
        "question": "Що показує довірчий інтервал для часткової кореляції?",
        "answer": (
            "Довірчий інтервал [r_lo, r_hi] — інтервал, який з імовірністю\n"
            "1−α = 0.95 накриває ІСТИННЕ значення коефіцієнта кореляції\n"
            "у генеральній сукупності.\n\n"
            "Будується ТІЛЬКИ для значущих коефіцієнтів!\n"
            "Для незначущих — немає сенсу (якщо r статистично = 0).\n\n"
            "Алгоритм (перетворення Фішера):\n"
            "  1) z = arctanh(r) = 0.5·ln[(1+r)/(1−r)]\n"
            "  2) SE_z = 1/sqrt(df−1)\n"
            "  3) ДІ для z:  z ± z_α/2 · SE_z\n"
            "  4) Повертаємо назад:  [tanh(z_lo), tanh(z_hi)]\n\n"
            "Перетворення Фішера потрібне бо розподіл r несиметричний,\n"
            "а z вже розподілений нормально — можна будувати симетричний ДІ."
        ),
        "formula": "z = arctanh(r),  ДІ: [tanh(z − z_α/2·SE), tanh(z + z_α/2·SE)]",
        "tag": "довірчий інтервал",
    },
    {
        "id": "06",
        "short": "Що означає α = 0.05?",
        "question": "Що означає рівень значущості α = 0.05?",
        "answer": (
            "α = 0.05 — імовірність ПОМИЛКОВО відхилити нульову гіпотезу H₀\n"
            "(помилка першого роду, або «хибна тривога»).\n\n"
            "H₀: коефіцієнт кореляції = 0 (зв'язку немає)\n"
            "H₁: коефіцієнт кореляції ≠ 0 (зв'язок є)\n\n"
            "Якщо α = 0.05 — ми допускаємо що в 5% випадків помилково\n"
            "визнаємо зв'язок значущим, коли він насправді випадковий.\n\n"
            "Довірчий рівень = 1 − α = 0.95\n\n"
            "Як сказав викладач:\n"
            "  «з імовірністю 0.95 істинне значення потрапить у цей інтервал»"
        ),
        "formula": "Рівень довіри = 1 − α = 0.95",
        "tag": "значущість",
    },
    {
        "id": "07",
        "short": "t-тест значущості",
        "question": "Як перевіряється значущість через t-тест?",
        "answer": (
            "Крок 1. Формулюємо гіпотези:\n"
            "  H₀: r = 0  (зв'язку немає)\n"
            "  H₁: r ≠ 0  (зв'язок є)\n\n"
            "Крок 2. Обчислюємо t-статистику:\n\n"
            "  Для парних (df = n−2):\n"
            "    t = r·sqrt(n−2) / sqrt(1−r²)\n\n"
            "  Для часткових (df = n−p):\n"
            "    t = r_part·sqrt(n−p) / sqrt(1−r_part²)\n\n"
            "Крок 3. Порівнюємо |t| з критичним t_крит = t(α/2, df)\n\n"
            "  |t| > t_крит  →  відхиляємо H₀, зв'язок ЗНАЧУЩИЙ  ✔\n"
            "  |t| ≤ t_крит  →  H₀ не відхиляємо, зв'язок незначущий  ✘\n\n"
            "Або еквівалентно: p-value < α → значущий"
        ),
        "formula": "t = r·sqrt(n−2)/sqrt(1−r²)   ~   t(n−2)",
        "tag": "t-тест",
    },
    {
        "id": "08",
        "short": "Коваріація",
        "question": "Що таке коваріація і чим відрізняється від кореляції?",
        "answer": (
            "Коваріація — міра спільної мінливості двох ознак:\n\n"
            "  cov(X,Y) = (1/(n−1)) · Σ(xᵢ−x̄)(yᵢ−ȳ)\n\n"
            "Недолік: залежить від одиниць виміру!\n"
            "  (якщо X в метрах, Y в кілограмах — cov у м·кг)\n\n"
            "Кореляція — нормована коваріація:\n\n"
            "  r_XY = cov(X,Y) / (s_X · s_Y)\n\n"
            "  • Безрозмірна величина\n"
            "  • Завжди від −1 до +1\n"
            "  • Не залежить від одиниць виміру\n\n"
            "Зв'язок: r — це cov, поділена на добуток СКВ обох ознак."
        ),
        "formula": "r_XY = cov(X,Y) / (sX · sY)",
        "tag": "кореляція",
    },
    {
        "id": "09",
        "short": "Парна vs Часткова кореляція",
        "question": "Чим відрізняється парна від часткової кореляції?",
        "answer": (
            "ПАРНА кореляція r_jk:\n"
            "  • Зв'язок між двома ознаками БЕЗ врахування інших\n"
            "  • Може бути ХИБНОЮ (через вплив третіх змінних)\n"
            "  • df = n − 2\n\n"
            "ЧАСТКОВА кореляція r_jk·rest:\n"
            "  • Зв'язок між двома ознаками, виключивши вплив ВСІХ інших\n"
            "  • Показує «чистий» зв'язок\n"
            "  • df = n − p\n\n"
            "Класичний приклад хибної кореляції:\n"
            "  Кількість пожежників ↔ Збиток від пожежі\n"
            "  Парна кореляція ВИСОКА — але обидва залежать від розміру пожежі!\n"
            "  Часткова (виключивши розмір пожежі) ≈ 0\n\n"
            "Обчислення часткової через precision matrix R⁻¹:\n"
            "  r_jk·rest = −R⁻¹_jk / sqrt(R⁻¹_jj · R⁻¹_kk)"
        ),
        "formula": "r_jk·rest = −R⁻¹_jk / sqrt(R⁻¹_jj · R⁻¹_kk)",
        "tag": "часткова кореляція",
    },
    {
        "id": "10",
        "short": "Кореляція між X1 і X2",
        "question": "Що означає конкретне значення r₁₂?",
        "answer": (
            "r₁₂ — коефіцієнт лінійної кореляції Пірсона між ознаками X1 і X2.\n\n"
            "Інтерпретація значень:\n"
            "  |r| < 0.3   — слабкий зв'язок\n"
            "  0.3–0.5    — помірний зв'язок\n"
            "  0.5–0.7    — суттєвий зв'язок\n"
            "  0.7–0.9    — сильний зв'язок\n"
            "  |r| > 0.9   — дуже сильний зв'язок\n\n"
            "  r > 0  →  прямий зв'язок (X1 росте — X2 росте)\n"
            "  r < 0  →  обернений зв'язок (X1 росте — X2 спадає)\n"
            "  r = 0  →  лінійного зв'язку немає\n\n"
            "ВАЖЛИВО: r вимірює тільки ЛІНІЙНИЙ зв'язок!\n"
            "Нелінійний зв'язок може бути при r ≈ 0."
        ),
        "formula": "r₁₂ = Σ(x_i1−x̄₁)(x_i2−x̄₂) / sqrt[Σ(x_i1−x̄₁)²·Σ(x_i2−x̄₂)²]",
        "tag": "кореляція",
    },
    {
        "id": "11",
        "short": "Довірчий інтервал 0.95",
        "question": "Що означає довірчий інтервал 0.95?",
        "answer": (
            "Як сказав викладач:\n"
            "  «З імовірністю 0.95 істинне значення параметра\n"
            "   потрапить у цей інтервал»\n\n"
            "Точніше: якщо повторити експеримент 100 разів —\n"
            "приблизно 95 разів побудований інтервал накриє\n"
            "справжнє значення параметра у генеральній сукупності.\n\n"
            "Ширина інтервалу залежить від:\n"
            "  • n — більше даних → вужчий інтервал\n"
            "  • α — менший α (0.01) → ширший інтервал\n"
            "  • s — більша варіативність → ширший інтервал\n\n"
            "При α=0.05:  рівень довіри = 1−0.05 = 0.95  →  95%\n"
            "При α=0.01:  рівень довіри = 1−0.01 = 0.99  →  99%"
        ),
        "formula": "Рівень довіри = 1 − α = 0.95",
        "tag": "довірчий інтервал",
    },
    {
        "id": "12",
        "short": "Інтервальне розбиття (кількість класів M)",
        "question": "Як визначається кількість класів гістограми?",
        "answer": (
            "Формула з посібника:\n\n"
            "При N < 100:\n"
            "  M = floor(√N),   якщо floor(√N) непарне\n"
            "  M = floor(√N)−1, якщо floor(√N) парне\n\n"
            "При N ≥ 100:\n"
            "  M = floor(∛N),   якщо floor(∛N) непарне\n"
            "  M = floor(∛N)−1, якщо floor(∛N) парне\n\n"
            "Приклади:\n"
            "  N=30:  floor(√30)=5 (непарне)  →  M=5\n"
            "  N=50:  floor(√50)=7 (непарне)  →  M=7\n"
            "  N=100: floor(∛100)=4 (парне)   →  M=3\n"
            "  N=200: floor(∛200)=5 (непарне) →  M=5\n"
            "  N=1000:floor(∛1000)=10(парне)  →  M=9\n\n"
            "Бажано непарне M — щоб центральний клас був симетричним."
        ),
        "formula": "N<100: M=floor(√N)  |  N≥100: M=floor(∛N)  (непарне)",
        "tag": "гістограма",
    },
    {
        "id": "13",
        "short": "Що якби N=10?",
        "question": "Що було б якби мали тільки 10 спостережень?",
        "answer": (
            "N=10 < 100  →  M = floor(√10) = 3  (непарне)  →  M = 3 класи\n\n"
            "Але головна проблема — кореляція НЕНАДІЙНА при n=10:\n\n"
            "Для парних кореляцій:\n"
            "  df = n−2 = 8\n"
            "  r_крит ≈ 0.632  (при α=0.05)\n"
            "  Тобто |r| має перевищити 0.63 щоб вважатись значущим!\n\n"
            "Для часткових кореляцій (p=3):\n"
            "  df = n−p = 10−3 = 7  — ще менше\n"
            "  r_крит ще більший\n\n"
            "Для множинної кореляції:\n"
            "  df1 = p−1 = 2,  df2 = n−p = 7\n\n"
            "Висновок: при n=10 більшість коефіцієнтів виявляться\n"
            "незначущими навіть якщо реальний зв'язок існує.\n"
            "Мінімально рекомендована вибірка: n > 30."
        ),
        "formula": "r_крит = t_крит / sqrt(n−2 + t_крит²)",
        "tag": "розмір вибірки",
    },
    {
        "id": "14",
        "short": "Що таке R (множинна кореляція)?",
        "question": "Що таке множинний коефіцієнт кореляції R?",
        "answer": (
            "R_j — кореляція між ознакою j та ЛІНІЙНОЮ КОМБІНАЦІЄЮ\n"
            "всіх інших ознак. Показує наскільки добре ознаку j\n"
            "можна передбачити через решту.\n\n"
            "Обчислення через precision matrix R⁻¹:\n\n"
            "  R²_j = 1 − 1/R⁻¹_jj\n"
            "  R_j  = sqrt(R²_j)  ∈ [0, 1]\n\n"
            "Інтерпретація:\n"
            "  R_j ≈ 0  →  ознака j незалежна від решти\n"
            "  R_j ≈ 1  →  j майже повністю визначається іншими\n"
            "              (можлива мультиколінеарність!)\n\n"
            "R² = частка дисперсії j, пояснена іншими ознаками.\n\n"
            "F-тест значущості:\n"
            "  F = [R²/(p−1)] / [(1−R²)/(n−p)]  ~  F(p−1, n−p)\n\n"
            "Зв'язок часткової та множинної через R⁻¹:\n"
            "  Обидві обчислюються з ОДНОЇ оберненої матриці R⁻¹"
        ),
        "formula": "R²_j = 1 − 1/R⁻¹_jj      F = [R²/(p−1)] / [(1−R²)/(n−p)]",
        "tag": "множинна кореляція",
    },
]

# Кольори тегів
TAG_COLORS = {
    "кореляція":         "#4f8ef7",
    "первинний аналіз":  "#4ecb88",
    "часткова кореляція":"#f5c542",
    "значущість":        "#e05c7a",
    "t-тест":            "#e05c7a",
    "довірчий інтервал": "#a78bfa",
    "гістограма":        "#4ecb88",
    "розмір вибірки":    "#f97316",
    "множинна кореляція":"#4f8ef7",
}


# ══════════════════════════════════════════════════════════════
class TabTheory(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app      = app
        self._current = None
        self._build()

    def _build(self):
        # ── Заголовок ──────────────────────────────────────────
        top = tk.Frame(self, bg=C['bg'], pady=10, padx=16)
        top.pack(fill='x')
        tk.Label(top,
                 text="📖  Теорія для захисту лабораторної роботи",
                 bg=C['bg'], fg=C['accent'],
                 font=('Consolas', 13, 'bold')).pack(side='left')
        tk.Label(top,
                 text=f"  {len(QUESTIONS)} питань",
                 bg=C['bg'], fg=C['subtext'],
                 font=('Consolas', 9)).pack(side='left', padx=6)

        ttk.Separator(self).pack(fill='x')

        # ── Основний PanedWindow ────────────────────────────────
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=0, pady=0)

        left  = tk.Frame(paned, bg=C['bg'])
        right = tk.Frame(paned, bg=C['panel'])
        paned.add(left,  weight=1)
        paned.add(right, weight=3)

        # ── ЛІВО: список питань ────────────────────────────────
        tk.Label(left, text="  Питання", bg=C['header_bg'],
                 fg=C['accent'], font=('Consolas', 9, 'bold'),
                 anchor='w', pady=6).pack(fill='x')

        # Пошук
        search_frame = tk.Frame(left, bg=C['bg'], pady=4, padx=6)
        search_frame.pack(fill='x')
        tk.Label(search_frame, text="🔍", bg=C['bg'],
                 fg=C['subtext'], font=('Consolas', 10)).pack(side='left')
        self._search_var = tk.StringVar()
        self._search_var.trace_add('write', self._on_search)
        search_entry = tk.Entry(
            search_frame, textvariable=self._search_var,
            bg=C['panel2'], fg=C['text'],
            insertbackground=C['accent'],
            relief='flat', font=('Consolas', 9), bd=1)
        search_entry.pack(side='left', fill='x', expand=True, padx=4)

        # Список
        list_frame = tk.Frame(left, bg=C['bg'])
        list_frame.pack(fill='both', expand=True)

        vsb = ttk.Scrollbar(list_frame, orient='vertical')
        self._listbox = tk.Listbox(
            list_frame,
            bg=C['panel'], fg=C['text'],
            selectbackground=C['accent'], selectforeground='white',
            font=('Consolas', 9), relief='flat',
            activestyle='none', bd=0,
            yscrollcommand=vsb.set)
        vsb.configure(command=self._listbox.yview)
        vsb.pack(side='right', fill='y')
        self._listbox.pack(fill='both', expand=True)
        self._listbox.bind('<<ListboxSelect>>', self._on_select)

        self._all_items = [(q['id'], q['short'], q) for q in QUESTIONS]
        self._populate_list(self._all_items)

        # ── ПРАВО: відповідь ──────────────────────────────────
        # Шапка питання
        self._q_header = tk.Frame(right, bg=C['header_bg'])
        self._q_header.pack(fill='x')

        self._q_num = tk.Label(self._q_header, text="",
                               bg=C['header_bg'], fg=C['subtext'],
                               font=('Consolas', 8), padx=12, pady=4)
        self._q_num.pack(side='left')

        self._q_tag_lbl = tk.Label(self._q_header, text="",
                                   bg=C['header_bg'], fg=C['yellow'],
                                   font=('Consolas', 8, 'bold'), padx=8)
        self._q_tag_lbl.pack(side='right')

        self._q_title = tk.Label(right, text="← Оберіть питання зі списку",
                                 bg=C['panel'], fg=C['accent'],
                                 font=('Consolas', 12, 'bold'),
                                 anchor='w', padx=16, pady=10, wraplength=700)
        self._q_title.pack(fill='x')

        ttk.Separator(right).pack(fill='x')

        # Формула
        self._formula_frame = tk.Frame(right, bg=C['bg'], pady=0)
        self._formula_lbl = tk.Label(
            self._formula_frame, text="",
            bg=C['panel2'], fg=C['yellow'],
            font=('Consolas', 11, 'bold'),
            anchor='w', padx=16, pady=10,
            wraplength=700, justify='left')
        self._formula_lbl.pack(fill='x', padx=12, pady=8)

        # Відповідь
        ans_frame = tk.Frame(right, bg=C['panel'])
        ans_frame.pack(fill='both', expand=True)

        vsb2 = ttk.Scrollbar(ans_frame, orient='vertical')
        self._answer_text = tk.Text(
            ans_frame,
            bg=C['panel'], fg=C['text'],
            font=('Consolas', 10),
            relief='flat', bd=0,
            wrap='word', padx=20, pady=14,
            state='disabled', cursor='arrow',
            yscrollcommand=vsb2.set,
            spacing1=2, spacing3=4)
        vsb2.configure(command=self._answer_text.yview)
        vsb2.pack(side='right', fill='y')
        self._answer_text.pack(fill='both', expand=True)

        # Теги для тексту
        self._answer_text.tag_configure('keyword',
            foreground=C['accent'], font=('Consolas', 10, 'bold'))
        self._answer_text.tag_configure('formula_inline',
            foreground=C['yellow'], font=('Consolas', 10, 'bold'))
        self._answer_text.tag_configure('check',
            foreground=C['green'], font=('Consolas', 10, 'bold'))
        self._answer_text.tag_configure('cross',
            foreground=C['accent2'])
        self._answer_text.tag_configure('note',
            foreground=C['subtext'], font=('Consolas', 9, 'italic'))

        # Нижня панель навігації
        nav = tk.Frame(right, bg=C['header_bg'], pady=6)
        nav.pack(fill='x', side='bottom')
        ttk.Button(nav, text="◀  Попереднє",
                   command=self._prev).pack(side='left', padx=12)
        self._nav_label = tk.Label(nav, text="", bg=C['header_bg'],
                                   fg=C['subtext'], font=('Consolas', 8))
        self._nav_label.pack(side='left', expand=True)
        ttk.Button(nav, text="Наступне  ▶",
                   command=self._next).pack(side='right', padx=12)

        # Показати перше питання
        if QUESTIONS:
            self._listbox.selection_set(0)
            self._show(QUESTIONS[0])

    # ── СПИСОК ────────────────────────────────────────────────
    def _populate_list(self, items):
        self._listbox.delete(0, 'end')
        for (qid, short, _) in items:
            self._listbox.insert('end', f"  {qid}. {short}")

    def _on_search(self, *_):
        q = self._search_var.get().lower().strip()
        if not q:
            filtered = self._all_items
        else:
            filtered = [
                item for item in self._all_items
                if q in item[1].lower()
                or q in item[2]['answer'].lower()
                or q in item[2]['tag'].lower()
                or q in item[2]['question'].lower()
            ]
        self._populate_list(filtered)
        self._filtered = filtered

    def _on_select(self, _evt=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        src = getattr(self, '_filtered', self._all_items)
        if idx < len(src):
            self._show(src[idx][2])

    # ── ВІДОБРАЖЕННЯ ПИТАННЯ ──────────────────────────────────
    def _show(self, q: dict):
        self._current = q

        # Заголовок
        self._q_num.configure(text=f"Питання #{q['id']}")
        tag_color = TAG_COLORS.get(q['tag'], C['yellow'])
        self._q_tag_lbl.configure(text=f"[ {q['tag']} ]",
                                   fg=tag_color)
        self._q_title.configure(text=q['question'])

        # Формула
        if q.get('formula'):
            self._formula_frame.pack(fill='x')
            self._formula_lbl.configure(text=f"  📐  {q['formula']}")
        else:
            self._formula_frame.pack_forget()

        # Відповідь
        self._answer_text.configure(state='normal')
        self._answer_text.delete('1.0', 'end')
        self._render_answer(q['answer'])
        self._answer_text.configure(state='disabled')

        # Навігація
        all_ids = [item[0] for item in self._all_items]
        try:
            pos = all_ids.index(q['id'])
            self._nav_label.configure(
                text=f"{pos+1} / {len(all_ids)}")
        except ValueError:
            pass

    def _render_answer(self, text: str):
        """Простий рендер тексту з підсвічуванням ключових слів."""
        HIGHLIGHTS = {
            '✔': 'check', '✘': 'cross',
        }
        KEYWORD_MARKERS = [
            'ВАЖЛИВО', 'ТІЛЬКИ', 'Висновок', 'ХИБНОЮ',
            'Як сказав викладач', 'НЕНАДІЙНА',
        ]
        for line in text.split('\n'):
            # Підсвічуємо ключові слова
            tagged = False
            for kw in KEYWORD_MARKERS:
                if kw in line:
                    self._answer_text.insert('end', line + '\n', 'keyword')
                    tagged = True
                    break
            if tagged:
                continue

            # Підсвічуємо ✔ і ✘
            if '✔' in line:
                self._answer_text.insert('end', line + '\n', 'check')
            elif '✘' in line:
                self._answer_text.insert('end', line + '\n', 'cross')
            elif line.strip().startswith('«') or line.strip().startswith('»'):
                self._answer_text.insert('end', line + '\n', 'formula_inline')
            else:
                self._answer_text.insert('end', line + '\n')

    # ── НАВІГАЦІЯ ─────────────────────────────────────────────
    def _prev(self):
        self._navigate(-1)

    def _next(self):
        self._navigate(+1)

    def _navigate(self, direction: int):
        if not self._current:
            return
        all_ids = [item[0] for item in self._all_items]
        try:
            pos = all_ids.index(self._current['id'])
        except ValueError:
            return
        new_pos = (pos + direction) % len(all_ids)
        self._listbox.selection_clear(0, 'end')
        self._listbox.selection_set(new_pos)
        self._listbox.see(new_pos)
        self._show(self._all_items[new_pos][2])
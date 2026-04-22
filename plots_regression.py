"""
plots_regression.py — Графіки для регресії та візуалізації
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from plots import _style_fig, _style_ax, BG, PANEL, TEXT, GRID, ACCENT, ACCENT2

def regression_diagnostic_figure(res: dict) -> Figure:
    fig = Figure(figsize=(10, 4), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    _style_ax(ax, "Діагностична діаграма (залишки vs прогнозовані значення)",
              "Прогнозовані значення ŷ", "Залишки e")

    ax.scatter(res['y_hat'], res['residuals'], s=15, color=ACCENT, alpha=0.7)
    ax.axhline(0, color=ACCENT2, linestyle='--', lw=1)
    ax.set_facecolor(PANEL)
    fig.tight_layout()
    return fig

def parallel_coordinates_figure(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(8, 5), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    _style_ax(ax, "Паралельні координати", "", "")

    for i, row in df.iterrows():
        ax.plot(range(df.shape[1]), row, marker='o', alpha=0.6, lw=1)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, color=TEXT)
    fig.tight_layout()
    return fig

def bubble_chart_figure(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 5), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    _style_ax(ax, "Бульбашкова діаграма (X1, X2, розмір = X3)", df.columns[0], df.columns[1])

    sizes = (df.iloc[:, 2] - df.iloc[:, 2].min()) / (df.iloc[:, 2].max() - df.iloc[:, 2].min()) * 300 + 20
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=sizes, alpha=0.7, c=df.iloc[:, 2], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label=df.columns[2])
    fig.tight_layout()
    return fig

def heatmap_data_figure(df: pd.DataFrame) -> Figure:
    """Теплова карта спостережень × ознаки (нормовані значення)"""
    from matplotlib.colors import Normalize
    from stats_regression import normalize_dataframe  # імпортуй з plots або з stats_regression
    
    df_norm = normalize_dataframe(df)
    
    fig = Figure(figsize=(10, 6), dpi=110)
    _style_fig(fig)
    ax = fig.add_subplot(111)
    
    im = ax.imshow(df_norm.T, aspect='auto', cmap='YlGn', interpolation='nearest')
    ax.set_xlabel('Спостереження', color=TEXT)
    ax.set_ylabel('Ознаки', color=TEXT)
    ax.set_title('Теплова карта даних (нормовані значення 0–1)', color=TEXT, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, color=TEXT, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Нормоване значення', color=TEXT)
    
    fig.tight_layout()
    return fig
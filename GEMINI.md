# 📊 Lab №5 — Multidimensional Data Analysis

## Project Overview
This project is a GUI-based Python application designed for complex multidimensional statistical analysis. It was developed as part of "Data Analysis" Lab №5. The application provides a comprehensive suite of tools for primary statistics, correlation analysis (pair, partial, multiple), regression models, and Principal Component Analysis (PCA), along with advanced data visualization.

### Key Technologies
- **Python 3.x**
- **Tkinter (ttk):** Modern dark-themed GUI architecture.
- **Pandas & NumPy:** Core data manipulation and matrix algebra.
- **SciPy:** Advanced statistical computations (t-tests, F-tests, Chi-square).
- **Matplotlib:** Rich data visualization and plotting.

### Architecture
The project follows a modular functional design:
- `main.py`: Entry point and GUI orchestration (tabs, event handling).
- `stats_core.py`, `stats_pca.py`, `stats_regression.py`: Computational kernels for specific statistical methods.
- `plots.py`, `plots_pca.py`, `plots_regression.py`: Visualization logic for generating Matplotlib figures.
- `gui_widgets.py`: Reusable UI components (tables, plot panels, cards) and theme configuration.

## Building and Running

### Prerequisites
Ensure you have Python installed with the following dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

### Running the Application
Execute the following command in the project root:
```bash
python main.py
```

### Usage Workflow
1. **Load Data:** Use the "📂 Дані" tab to import a text file (e.g., `norm3n.txt`).
2. **Analysis Tabs:** Navigate through tabs to perform specific analyses:
   - **📈 Первинний аналіз:** Basic statistics and histograms.
   - **🔗 Парні кореляції:** Pearson correlation matrices and heatmaps.
   - **⚙️ Часткові кореляції:** Partial correlation analysis with Forest Plots.
   - **🔢 Множинні кореляції:** Multiple correlation coefficients.
   - **📐 Множинна регресія:** Linear regression models with diagnostic plots.
   - **📊 Візуалізація:** Advanced plots like Parallel Coordinates, Glyphs, and Star Plots.
   - **PCA (TabPCA):** Principal Component Analysis (standardization, eigenvalues, variance).

## Development Conventions

### Coding Style
- **Naming:** Follows standard Python (snake_case) for functions/variables and PascalCase for UI classes.
- **Documentation:** Most files contain header docstrings explaining their purpose and specific statistical algorithms used.
- **Theming:** All UI components use the centralized `C` color palette defined in `gui_widgets.py` for a consistent "Dark Academic" look.

### Testing Practices
- **TODO:** Currently, verification is done manually through the GUI. Implementing unit tests for the `stats_*.py` kernels using `pytest` is recommended for future development.

### UI Consistency
- Use `PlotPanel` for embedding Matplotlib figures.
- Use `DataTable` for displaying tabular data.
- Status updates should be directed through the `StatusBar` (e.g., `self.app.status.ok()`).

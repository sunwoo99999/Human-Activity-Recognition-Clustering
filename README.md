# HAR HAR Dataset K-Means Clustering Analysis Program

This Python program analyzes the Human Activity Recognition (HAR) dataset using K-Means clustering and verifies statistical significance of results.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Output](#output)

## Project Overview

The program performs the following steps on the Kaggle HAR dataset:

1. Data preprocessing: missing value handling, outlier removal, normalization, PCA
2. Exploratory data analysis (EDA): basic statistics, correlations, feature importance
3. K-Means clustering: search for optimal K and cluster analysis
4. Evaluation: compute internal and external clustering metrics
5. Statistical testing: ANOVA, Kruskal–Wallis, effect size calculations
6. Visualization: Elbow curve, Silhouette plot, t-SNE, etc.
7. Report generation: Markdown report and CSV tables

## ✨ Key Features

### Data preprocessing

- Automatic missing value handling (drop/mean/median)
- Outlier removal using IQR
- Normalization with `StandardScaler` or `MinMaxScaler`
- Optional dimensionality reduction via PCA

### Clustering

- Apply K-Means algorithm
- Automatic search for optimal K (Elbow, Silhouette, etc.)
- Per-cluster statistical analysis

### Evaluation metrics

- Internal metrics: Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index, Inertia
- External metrics: Adjusted Rand Index, NMI, Homogeneity, Completeness, V-measure, Purity

### Statistical tests

- One-way ANOVA
- Kruskal–Wallis test
- Tukey HSD post-hoc test
- Cohen's d effect size
- Eta-squared (η²)
- Multiple testing correction (Bonferroni)

### Visualization

- Elbow curve
- Silhouette score trend
- 2D cluster visualization (PCA, t-SNE)
- Silhouette diagram
- Confusion matrix heatmap
- Feature importance bar chart
- Correlation heatmap
- ANOVA p-value distribution

#### Key Visualizations

**Elbow Curve & Silhouette Analysis**

![Elbow Curve](results/figures/elbow_curve.png)

![Silhouette Scores](results/figures/silhouette_scores.png)

**Cluster Visualization (2D Projection)**

![PCA Clusters](results/figures/clusters_2d_pca.png)

![t-SNE Clusters](results/figures/clusters_2d_tsne.png)

**Silhouette Diagram**

![Silhouette Diagram](results/figures/silhouette_diagram.png)

## Usage

### Basic run

```bash
python main.py
```

The program runs the full pipeline automatically:

1. Data loading and preprocessing
2. EDA
3. K-Means clustering
4. Evaluation and statistical testing
5. Visualization
6. Report generation

### Runtime

- Typical runtime: 5–10 minutes (depends on dataset size)
- t-SNE visualization is the most time-consuming step

## Project structure

```
code/
├── main.py                    # main runner
├── config.py                  # configuration file
├── utils.py                   # utility functions
├── data_loader.py             # data loading and preprocessing
├── eda.py                     # exploratory data analysis
├── clustering.py              # K-Means clustering
├── evaluation.py              # evaluation metrics
├── statistical_tests.py       # statistical tests
├── visualization.py           # visualization
├── report_generator.py        # report generation
├── requirements.txt           # dependency list
├── README.md                  # this file
├── train.csv                  # training data (prepare as needed)
├── test.csv                   # test data (prepare as needed)
└── results/                   # output directory (auto-created)
  ├── figures/               # graph images
  ├── tables/                # CSV tables
  └── logs/                  # log files
```

## Configuration

Modify `config.py` to change runtime behavior. Key settings include:

```python
# Random seed (reproducibility)
RANDOM_SEED = 42

# Range of cluster counts to try
K_RANGE = list(range(2, 11))

# Normalization method: 'standard' or 'minmax'
NORMALIZATION_METHOD = 'standard'

# PCA usage
USE_PCA = True
PCA_VARIANCE_RATIO = 0.95

# Criterion for selecting optimal K
OPTIMAL_K_CRITERION = 'silhouette'

# Significance level
ALPHA_LEVEL = 0.05

# Figure DPI
FIGURE_DPI = 300

# Number of iterations for reproducibility checks
N_ITERATIONS = 10
```

Data settings:

```python
# Label column name
LABEL_COLUMN = 'Activity'

# Missing value strategy: 'drop', 'mean', 'median'
MISSING_VALUE_STRATEGY = 'drop'

# Outlier removal
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 3.0
```

## Output

### Figures (`results/figures/`)

**Clustering Evaluation Metrics**

![All Metrics](results/figures/all_metrics.png)

![Cluster Sizes](results/figures/cluster_sizes.png)

**Feature Analysis**

![Feature Importance](results/figures/feature_importance.png)

![Correlation Heatmap](results/figures/correlation_heatmap.png)

**Confusion Matrix & Statistical Tests**

![Confusion Matrix](results/figures/confusion_matrix.png)

![P-value Distribution](results/figures/pvalue_distribution.png)

### Tables (`results/tables/`)

- `clustering_results.csv` (metrics per K)
- `internal_metrics.csv`
- `external_metrics.csv`
- `anova_results.csv`
- `feature_importance.csv`

### Report (`results/`)

- `analysis_report.md`

### Logs (`results/logs/`)

- `analysis.log`

## Example summary

After running, the console prints a summary similar to:

```
================================================================================
                             Analysis Complete!
================================================================================

Data:
  - Training samples: 7,352
  - Test samples: 2,947
  - Number of features: 561

Clustering:
  - Selected K: 6
  - Silhouette Score: 0.4523
  - Davies-Bouldin Index: 0.8234
  - Calinski-Harabasz Index: 4523.45

External metrics:
  - Adjusted Rand Index: 0.6789
  - Normalized Mutual Information: 0.7234
  - Purity: 0.8012

Statistical testing:
  - Total features: 561
  - Significant features (ANOVA, p<0.05): 523
  - Significant feature ratio: 93.23%

Results saved to:
  - Figures: results\figures
  - Tables: results\tables
  - Report: results\analysis_report.md
  - Logs: results\logs\analysis.log
```

## Troubleshooting

### 1. Package installation issues

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### 2. Out of memory

Lower the sampling ratio in `config.py`:

```python
SAMPLING_RATIO = 0.5  # use 50% of samples
```

### 3. Korean font rendering issues (Windows)

Check the font settings in `visualization.py`:

```python
plt.rcParams['font.family'] = 'Malgun Gothic'  # Malgun Gothic
```

### 4. Label column errors

Update `LABEL_COLUMN` in `config.py` to match the actual label column name in your dataset.

## 📚 References

- **K-Means**: [scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- **Silhouette Score**: [scikit-learn Silhouette](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
- **ANOVA**: [scipy.stats.f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
- **t-SNE**: [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

## 📧 Support

If you encounter issues while using the program, check the log file (`results/logs/analysis.log`).

## 📄 License

This project may be used for research and educational purposes.

**Last updated**: 2026-01-19
**Created on**: 2026-01-19

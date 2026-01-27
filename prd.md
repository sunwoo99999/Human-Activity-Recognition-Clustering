# PRD: HAR Dataset K-Means Analysis Program

## 1. Project Overview

### 1.1 Purpose

Develop a Python analysis program that applies K-Means clustering to the
Kaggle Human Activity Recognition (HAR) dataset and performs statistical
validation to produce experimental results suitable for academic publication.

### 1.2 Background

- Human Activity Recognition (HAR) automatically classifies human activities
  from sensor data and is an active research area.
- It is important to evaluate the applicability of unsupervised methods
  (e.g., K-Means) to HAR data.
- Statistical validation is required to ensure the reliability of results.

### 1.3 Scope

- Data preprocessing and exploratory data analysis (EDA)
- Implementation and tuning of K-Means clustering
- Statistical significance testing (multiple test methods)
- Visualization and report generation

## 2. Dataset Information

### 2.1 Dataset structure

- **train.csv**: training dataset
- **test.csv**: test dataset
- Includes sensor-based features and activity labels

### 2.2 Expected features

- Accelerometer data
- Gyroscope data
- Time/frequency domain features
- Statistical features (mean, std, min/max, etc.)

## 3. Functional Requirements

### 3.1 Data preprocessing

- **FR-1.1**: Load CSV files and inspect data structure
- **FR-1.2**: Detect and handle missing values (impute or drop)
- **FR-1.3**: Detect and handle outliers
- **FR-1.4**: Normalize/standardize features (StandardScaler, MinMaxScaler)
- **FR-1.5**: Feature selection and dimensionality reduction (PCA)

### 3.2 Exploratory Data Analysis (EDA)

- **FR-2.1**: Compute basic statistics (mean, median, std, quartiles)
- **FR-2.2**: Analyze correlations between features
- **FR-2.3**: Inspect class distribution by activity type
- **FR-2.4**: Visualize reduced-dimension embeddings (PCA, t-SNE)
- **FR-2.5**: Plot histograms of key features

### 3.3 K-Means Clustering

- **FR-3.1**: Use scikit-learn K-Means implementation
- **FR-3.2**: Determine optimal K via:
  - Elbow Method
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
- **FR-3.3**: Test initialization methods (k-means++, random)
- **FR-3.4**: Save clustering results (cluster labels)
- **FR-3.5**: Analyze cluster centroids

### 3.4 Model Evaluation

- **FR-4.1**: Internal metrics
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Inertia (Within-Cluster Sum of Squares)
- **FR-4.2**: External metrics (when labels available)
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Homogeneity, Completeness, V-measure
  - Purity
- **FR-4.3**: Generate confusion matrix

### 3.5 Statistical Significance Testing

- **FR-5.1**: Test differences between clusters
  - ANOVA: test whether feature values differ across clusters
- **FR-5.2**: Reproducibility checks
  - Multiple runs with different random seeds (>=10 runs)
  - Compute mean and standard deviation of results
- **FR-5.3**: Compute and interpret p-values (α = 0.05)

### 3.6 Visualization

- **FR-6.1**: Cluster visualizations
  - 2D/3D scatter plots (PCA-reduced)
  - t-SNE plots
- **FR-6.2**: Metric visualizations
  - Elbow curve
  - Silhouette plots
  - Box plots of feature distributions by cluster
- **FR-6.3**: Statistical test visualizations
  - Heatmap of p-values
- **FR-6.4**: Confusion matrix heatmap

### 3.7 Output and Reporting

- **FR-7.1**: Summary report (text/Markdown)
- **FR-7.2**: Export statistical tables (CSV/Excel)
- **FR-7.3**: Save visualizations (PNG/PDF, high resolution)
- **FR-7.4**: Log experimental parameters and execution times
- **FR-7.5**: Save reproducible artifacts (models, random seeds)

## 4. Technical Requirements

### 4.1 Development environment

- **Python**: 3.8+
- **Development tools**: Jupyter Notebook or Python scripts

### 4.2 Required libraries

- **Data processing**: pandas, numpy
- **Machine learning**: scikit-learn
- **Statistical analysis**: scipy, statsmodels, pingouin
- **Visualization**: matplotlib, seaborn, plotly
- **Dimensionality reduction**: scikit-learn (PCA, t-SNE)
- **Utilities**: warnings, logging, datetime

### 4.3 Optional libraries

- **yellowbrick**: ML visualization
- **kneed**: automatic elbow detection
- **openpyxl**: Excel export

### 4.4 Code structure

```
code/
├── main.py                      # main runner
├── config.py                    # configuration (parameters)
├── data_loader.py               # data loading and preprocessing
├── eda.py                       # exploratory data analysis
├── clustering.py                # K-Means clustering
├── evaluation.py                # model evaluation
├── statistical_tests.py         # statistical tests
├── visualization.py             # visualization
├── report_generator.py          # report generation
├── utils.py                     # utility functions
├── requirements.txt             # dependency list
└── results/                     # output directory
    ├── figures/
    ├── tables/
    └── logs/
```

## 5. Non-functional Requirements

### 5.1 Performance

- **NFR-1.1**: Complete full analysis pipeline within ~10 minutes on a
  typical laptop (depends on dataset size)
- **NFR-1.2**: Support for large datasets with memory-efficient processing

### 5.2 Reproducibility

- **NFR-2.1**: Use fixed random seeds to enable reproducibility
- **NFR-2.2**: Log all parameters and settings
- **NFR-2.3**: Record version information (libraries, Python)

### 5.3 Readability & Maintainability

- **NFR-3.1**: Follow PEP 8 coding style
- **NFR-3.2**: Document functions/classes with docstrings
- **NFR-3.3**: Use comments to explain non-obvious code
- **NFR-3.4**: Modular structure for extensibility

### 5.4 Usability

- **NFR-4.1**: Allow parameter tuning via CLI or config file
- **NFR-4.2**: Provide progress indicators (progress bar)
- **NFR-4.3**: Clear error handling and informative messages

## 6. Experimental Design

### 6.1 Scenarios

1. **Baseline**: Run clustering for K = [2..10]
2. **Feature selection experiments**: Compare performance across feature subsets
3. **Preprocessing experiments**: Compare normalization methods
4. **Stability experiments**: Repeat experiments 10 times for reproducibility

### 6.2 Evaluation criteria

- Optimal K: maximize Silhouette Score
- Statistical significance: p-value < 0.05
- Cluster quality: minimize Davies-Bouldin Index
- Label agreement: NMI > 0.5

## 7. Deliverables

### 7.1 Code

- Modular Python analysis program
- Jupyter Notebook demonstrating the analysis
- requirements.txt

### 7.2 Outputs

- High-resolution visualization files (PNG/PDF)
- Statistical result tables (CSV/Excel)
- Analysis report (Markdown/PDF)
- Experiment log files

### 7.3 Documentation

- README.md (usage instructions)
- Docstrings in code
- Experiment summary report

## 8. Schedule & Milestones

### Phase 1: Environment & data preparation (1-2 days)

- Setup development environment
- Load dataset and run EDA

### Phase 2: Clustering implementation (2-3 days)

- Implement K-Means workflow
- Search for optimal K
- Compute baseline evaluation metrics

### Phase 3: Statistical validation (2-3 days)

- Implement statistical tests
- Compute effect sizes
- Reproducibility experiments

### Phase 4: Visualization & reporting (1-2 days)

- Generate visualizations
- Write report and documentation

### Phase 5: Validation & finalization (1 day)

- Code review and testing
- Final experiments and result consolidation

## 9. Success Criteria

### 9.1 Technical success

- ✅ Implement all required functionality
- ✅ Produce statistically significant results (p < 0.05)
- ✅ Reproducible experimental results
- ✅ High-quality visualizations

### 9.2 Research success

- ✅ Obtain analysis results sufficient for paper submission
- ✅ Achieve reasonable clustering performance (Silhouette > 0.3)
- ✅ Provide clear statistical evidence
- ✅ Produce publication-quality figures

## 10. Risks & Mitigations

| Risk                                   | Impact | Mitigation                                    |
| -------------------------------------- | ------ | --------------------------------------------- |
| Data quality issues (missing/outliers) | High   | Implement robust preprocessing pipeline       |
| K-Means poor performance               | Medium | Consider alternative algorithms (GMM, DBSCAN) |
| Lack of statistical significance       | High   | Apply multiple tests and effect size analyses |
| Excessive computation time             | Low    | Use sampling and parallelization              |
| Reproducibility issues                 | Medium | Fix random seeds and track versions           |

## 11. Notes

### 11.1 References

- Fundamentals of K-Means clustering
- Prior work on the HAR dataset
- Statistical validation methods (ANOVA, post-hoc tests)
- Interpretation of clustering evaluation metrics

### 11.2 Dataset links

- (Add specific dataset URLs as needed)

### 11.3 Additional considerations

- Check target journal requirements for statistical validation
- Consider Docker for reproducibility (optional)
- Consider interactive visualizations (Plotly)

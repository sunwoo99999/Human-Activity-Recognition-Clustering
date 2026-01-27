"""
config.py
Configuration for the HAR dataset K-Means analysis program.

This module centralizes all configuration values used across the analysis
pipeline. It defines random seeds, file paths, hyperparameters, and other
settings to ensure reproducibility and easy tuning.
"""

import os
from pathlib import Path

class Config:
    """
    Class that manages configuration for the analysis program.
    """
    
    # ==================== Base paths ====================
    # Project root directory (parent directory of this file)
    BASE_DIR = Path(__file__).parent
    
    # Data file paths
    TRAIN_DATA_PATH = BASE_DIR / "train.csv"
    TEST_DATA_PATH = BASE_DIR / "test.csv"
    
    # Results directories
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    LOGS_DIR = RESULTS_DIR / "logs"
    
    # ==================== Reproducibility ====================
    # Fix random seed for reproducible results
    RANDOM_SEED = 42
    
    # Number of repeated experiments (for stability checks)
    N_ITERATIONS = 10
    
    # ==================== Data preprocessing ====================
    # Label column name (modify to match your dataset)
    LABEL_COLUMN = 'Activity'  # or 'label', 'target', etc.
    
    # Normalization method: 'standard' or 'minmax'
    NORMALIZATION_METHOD = 'standard'
    
    # Missing value handling strategy: 'drop', 'mean', or 'median'
    MISSING_VALUE_STRATEGY = 'drop'
    
    # Outlier removal settings
    REMOVE_OUTLIERS = True  # remove extreme outliers only
    # Outlier detection threshold (multiple of IQR)
    OUTLIER_THRESHOLD = 12.0  # very lenient - aims to remove ~15-20%
    
    # ==================== Feature selection ====================
    # Whether to apply feature selection (ANOVA F-value based)
    USE_FEATURE_SELECTION = True
    # Number of top features to select (None to use p-value threshold)
    N_FEATURES_TO_SELECT = 50  # None or integer
    # Feature selection p-value threshold (used when N_FEATURES_TO_SELECT is None)
    FEATURE_SELECTION_PVALUE = 0.01
    
    # ==================== PCA settings ====================
    # Whether to apply PCA
    USE_PCA = True
    # PCA variance ratio to retain (0.95 = retain 95% variance)
    PCA_VARIANCE_RATIO = 0.95
    # Or use a fixed number of components
    PCA_N_COMPONENTS = None  # None will use variance_ratio
    
    # ==================== K-Means clustering ====================
    # Range of cluster counts to try
    K_RANGE = list(range(2, 11))  # [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # K-Means algorithm parameters
    KMEANS_INIT = 'k-means++'  # initialization method: 'k-means++' or 'random'
    KMEANS_MAX_ITER = 300  # maximum iterations
    KMEANS_N_INIT = 10  # number of runs with different centroid seeds
    KMEANS_TOL = 1e-4  # convergence tolerance
    
    # ==================== Evaluation settings ====================
    # Criterion to select optimal K: 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'composite'
    OPTIMAL_K_CRITERION = 'composite'
    
    # Composite score weights (used when OPTIMAL_K_CRITERION='composite')
    COMPOSITE_WEIGHTS = {
        'silhouette': 0.4,      # cluster cohesion
        'davies_bouldin': 0.2,  # cluster separation (lower is better)
        'purity': 0.4           # class purity
    }
    
    # ==================== Statistical testing ====================
    # Significance level (p-value threshold)
    ALPHA_LEVEL = 0.05
    
    # Post-hoc test method: 'tukey', 'bonferroni'
    POSTHOC_METHOD = 'tukey'
    
    # ==================== Visualization settings ====================
    # Plot style
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # matplotlib style
    
    # Figure size (inches)
    FIGURE_SIZE = (12, 8)
    
    # Resolution (DPI)
    FIGURE_DPI = 300
    
    # Image save format
    IMAGE_FORMAT = 'png'  # 'png', 'pdf', 'svg', etc.
    
    # Color palette
    COLOR_PALETTE = 'tab10'  # matplotlib colormap
    
    # t-SNE parameters
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    TSNE_LEARNING_RATE = 200
    
    # ==================== Logging ====================
    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_LEVEL = 'INFO'
    
    # Log file name
    LOG_FILE = 'analysis.log'
    
    # ==================== Report settings ====================
    # Report file name
    REPORT_FILE = 'analysis_report.md'
    
    # Results table format
    TABLE_FORMAT = 'csv'  # 'csv' or 'excel'
    
    # ==================== Performance ====================
    # Number of CPU cores to use for parallel processing (-1 uses all cores)
    N_JOBS = -1
    
    # Sampling ratio for memory efficiency (1.0 = use full dataset)
    SAMPLING_RATIO = 1.0
    
    @classmethod
    def create_directories(cls):
        """
        Create directories for storing results.

        This method creates all directories used to store analysis outputs.
        Existing directories are ignored.
        """
        directories = [
            cls.RESULTS_DIR,
            cls.FIGURES_DIR,
            cls.TABLES_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_log_path(cls):
        """
        Return the full path to the log file.

        Returns:
            Path: path to the log file
        """
        return cls.LOGS_DIR / cls.LOG_FILE
    
    @classmethod
    def get_report_path(cls):
        """
        Return the full path to the report file.

        Returns:
            Path: path to the report file
        """
        return cls.RESULTS_DIR / cls.REPORT_FILE
    
    @classmethod
    def summary(cls):
        """
        Return a dictionary summarizing key configuration values.

        Returns:
            dict: major configuration values
        """
        return {
            'random_seed': cls.RANDOM_SEED,
            'k_range': cls.K_RANGE,
            'normalization_method': cls.NORMALIZATION_METHOD,
            'use_pca': cls.USE_PCA,
            'pca_variance_ratio': cls.PCA_VARIANCE_RATIO,
            'kmeans_init': cls.KMEANS_INIT,
            'alpha_level': cls.ALPHA_LEVEL,
            'n_iterations': cls.N_ITERATIONS
        }


# Create configuration object (importable by other modules)
config = Config()

"""
data_loader.py
HAR dataset loading and preprocessing module

Provides functionality for loading CSVs, handling missing values,
removing outliers, normalization, PCA, and feature selection.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from config import Config
from utils import Timer, validate_dataframe


class DataLoader:
    """
    Class responsible for loading and preprocessing the HAR dataset.
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (Config, optional): configuration object. If None, uses default Config
        """
        self.config = config if config is not None else Config()
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.selected_feature_indices = None  # Feature selection result
        self.feature_scores = None  # Feature importance scores
    
    def load_data(self, train_path=None, test_path=None):
        """
        Load training and test data from CSV files.

        Args:
            train_path (str or Path, optional): path to training data
            test_path (str or Path, optional): path to test data

        Returns:
            tuple: (train_df, test_df)
        """
        if train_path is None:
            train_path = self.config.TRAIN_DATA_PATH
        if test_path is None:
            test_path = self.config.TEST_DATA_PATH
        
        with Timer("Data loading"):
            # Load CSV files
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            
            logging.info(f"Train data shape: {self.train_df.shape}")
            logging.info(f"Test data shape: {self.test_df.shape}")
            
            # Sampling (use fraction specified in config)
            if self.config.SAMPLING_RATIO < 1.0:
                n_train = int(len(self.train_df) * self.config.SAMPLING_RATIO)
                n_test = int(len(self.test_df) * self.config.SAMPLING_RATIO)
                self.train_df = self.train_df.sample(n=n_train, random_state=self.config.RANDOM_SEED)
                self.test_df = self.test_df.sample(n=n_test, random_state=self.config.RANDOM_SEED)
                logging.info(f"Sampled train data shape: {self.train_df.shape}")
                logging.info(f"Sampled test data shape: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def check_missing_values(self):
        """
        Check for missing values and return a report.

        Returns:
            dict: missing value information for each dataset
        """
        logging.info("Checking for missing values...")
        
        train_missing = self.train_df.isnull().sum()
        test_missing = self.test_df.isnull().sum()
        
        train_missing_count = train_missing.sum()
        test_missing_count = test_missing.sum()
        
        logging.info(f"Train missing values: {train_missing_count}")
        logging.info(f"Test missing values: {test_missing_count}")
        
        return {
            'train': train_missing[train_missing > 0],
            'test': test_missing[test_missing > 0]
        }
    
    def handle_missing_values(self, strategy=None):
        """
        Handle missing values according to the given strategy.

        Args:
            strategy (str, optional): strategy to handle missing values ('drop', 'mean', 'median')
        """
        if strategy is None:
            strategy = self.config.MISSING_VALUE_STRATEGY

        with Timer(f"Missing value handling ({strategy})"):
            if strategy == 'drop':
                # Drop rows containing any missing values
                self.train_df = self.train_df.dropna()
                self.test_df = self.test_df.dropna()
                logging.info("Dropped rows with missing values")

            elif strategy == 'mean':
                # Replace missing values with column mean
                numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
                self.train_df[numeric_cols] = self.train_df[numeric_cols].fillna(
                    self.train_df[numeric_cols].mean()
                )
                self.test_df[numeric_cols] = self.test_df[numeric_cols].fillna(
                    self.test_df[numeric_cols].mean()
                )
                logging.info("Filled missing values with mean")

            elif strategy == 'median':
                # Replace missing values with column median
                numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
                self.train_df[numeric_cols] = self.train_df[numeric_cols].fillna(
                    self.train_df[numeric_cols].median()
                )
                self.test_df[numeric_cols] = self.test_df[numeric_cols].fillna(
                    self.test_df[numeric_cols].median()
                )
                logging.info("Filled missing values with median")

            logging.info(f"Train shape after handling missing values: {self.train_df.shape}")
            logging.info(f"Test shape after handling missing values: {self.test_df.shape}")
    
    def remove_outliers(self, threshold=None):
        """
        Remove outliers using the IQR method.

        Args:
            threshold (float, optional): multiple of IQR to define outliers (default: 3.0)
        """
        if threshold is None:
            threshold = self.config.OUTLIER_THRESHOLD
        
        if not self.config.REMOVE_OUTLIERS:
            logging.info("Outlier removal is disabled in config")
            return
        
        with Timer("Outlier removal"):
            # Select numeric columns excluding the label column
            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            if self.config.LABEL_COLUMN in numeric_cols:
                numeric_cols = numeric_cols.drop(self.config.LABEL_COLUMN)
            
            # Compute IQR
            Q1 = self.train_df[numeric_cols].quantile(0.25)
            Q3 = self.train_df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            
            # Compute bounds for outlier detection
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Keep only rows that are not outliers
            mask = ~((self.train_df[numeric_cols] < lower_bound) | 
                     (self.train_df[numeric_cols] > upper_bound)).any(axis=1)
            
            original_size = len(self.train_df)
            self.train_df = self.train_df[mask]
            removed_count = original_size - len(self.train_df)
            
            logging.info(f"Removed {removed_count} outlier rows from train data")
            logging.info(f"Train shape after outlier removal: {self.train_df.shape}")
    
    def select_features(self, n_features=None, p_value_threshold=None):
        """
        Feature selection based on ANOVA F-value.

        Select features with large between-class differences to improve clustering quality.

        Args:
            n_features (int, optional): number of features to select. If None, use p-value threshold
            p_value_threshold (float, optional): p-value threshold used when n_features is None
        """
        if not self.config.USE_FEATURE_SELECTION:
            logging.info("Feature selection is disabled in config")
            return
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Features and labels must be separated first. Call separate_features_labels()")
        
        with Timer("Feature selection"):
            if n_features is None:
                n_features = self.config.N_FEATURES_TO_SELECT
            if p_value_threshold is None:
                p_value_threshold = self.config.FEATURE_SELECTION_PVALUE
            
            # Compute ANOVA F-scores and p-values
            f_scores, p_values = f_classif(self.X_train, self.y_train)
            
            # Feature selection
            if n_features is not None:
                # Select top n features
                top_indices = np.argsort(f_scores)[::-1][:n_features]
                self.selected_feature_indices = sorted(top_indices)
                logging.info(f"Selected top {n_features} features by F-score")
            else:
                # Select by p-value threshold
                significant_indices = np.where(p_values < p_value_threshold)[0]
                if len(significant_indices) == 0:
                    logging.warning(f"No features with p-value < {p_value_threshold}, using top 30")
                    top_indices = np.argsort(f_scores)[::-1][:30]
                    self.selected_feature_indices = sorted(top_indices)
                else:
                    self.selected_feature_indices = sorted(significant_indices)
                    logging.info(f"Selected {len(self.selected_feature_indices)} features with p < {p_value_threshold}")
            
            # Save feature scores
            self.feature_scores = pd.DataFrame({
                'feature': [self.feature_names[i] for i in range(len(f_scores))],
                'f_score': f_scores,
                'p_value': p_values
            }).sort_values('f_score', ascending=False)
            
            # Keep only selected features
            self.X_train = self.X_train[:, self.selected_feature_indices]
            self.X_test = self.X_test[:, self.selected_feature_indices]
            
            # Update feature names
            self.feature_names = [self.feature_names[i] for i in self.selected_feature_indices]
            
            logging.info(f"Feature selection completed: {len(self.selected_feature_indices)} features selected")
            logging.info(f"New shape: {self.X_train.shape}")
            logging.info(f"Top 5 features: {self.feature_names[:5]}")
    
    def separate_features_labels(self):
        """
        Separate features and labels.

        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        with Timer("Separating features and labels"):
            label_col = self.config.LABEL_COLUMN

            # Check whether the label column exists
            if label_col in self.train_df.columns:
                self.y_train = self.train_df[label_col].values
                self.y_test = self.test_df[label_col].values

                # Extract features (exclude label)
                self.X_train = self.train_df.drop(columns=[label_col]).values
                self.X_test = self.test_df.drop(columns=[label_col]).values

                # Save feature names
                self.feature_names = self.train_df.drop(columns=[label_col]).columns.tolist()

                logging.info(f"Features shape: {self.X_train.shape}")
                logging.info(f"Labels - unique classes: {np.unique(self.y_train)}")
            else:
                # If no label column, use all columns as features
                logging.warning(f"Label column '{label_col}' not found. Using all columns as features")
                self.X_train = self.train_df.values
                self.X_test = self.test_df.values
                self.feature_names = self.train_df.columns.tolist()
                self.y_train = None
                self.y_test = None
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def normalize_data(self, method=None):
        """
        Data normalization/standardization.

        Args:
            method (str, optional): normalization method ('standard' or 'minmax')

        Returns:
            tuple: (X_train_normalized, X_test_normalized)
        """
        if method is None:
            method = self.config.NORMALIZATION_METHOD
        
        with Timer(f"Data normalization ({method})"):
            if method == 'standard':
                # Standardization: mean 0, std 1
                self.scaler = StandardScaler()
            elif method == 'minmax':
                # Min-Max scaling: scale features to [0, 1]
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Fit scaler on training data then transform
            self.X_train = self.scaler.fit_transform(self.X_train)
            # Transform test data using the fitted scaler
            self.X_test = self.scaler.transform(self.X_test)
            
            logging.info(f"Data normalized using {method} method")
            logging.info(f"Train mean: {self.X_train.mean():.4f}, std: {self.X_train.std():.4f}")
        
        return self.X_train, self.X_test
    
    def apply_pca(self, n_components=None, variance_ratio=None):
        """
        Dimensionality reduction via Principal Component Analysis (PCA).

        Args:
            n_components (int, optional): number of components to keep
            variance_ratio (float, optional): variance ratio to retain (0.95 = 95%)

        Returns:
            tuple: (X_train_pca, X_test_pca, pca)
        """
        if not self.config.USE_PCA:
            logging.info("PCA is disabled in config")
            return self.X_train, self.X_test, None
        
        # Parameter setup
        if n_components is None:
            n_components = self.config.PCA_N_COMPONENTS
        if variance_ratio is None:
            variance_ratio = self.config.PCA_VARIANCE_RATIO
        
        with Timer("PCA transformation"):
            # If n_components is not specified, use variance_ratio
            if n_components is None:
                self.pca = PCA(n_components=variance_ratio, random_state=self.config.RANDOM_SEED)
            else:
                self.pca = PCA(n_components=n_components, random_state=self.config.RANDOM_SEED)
            
            # Apply PCA
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_test = self.pca.transform(self.X_test)
            
            # Explained variance information
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            logging.info(f"PCA reduced dimensions to {self.pca.n_components_}")
            logging.info(f"Explained variance ratio: {cumulative_variance[-1]:.4f}")
            logging.info(f"New shape: {self.X_train.shape}")
        
        return self.X_train, self.X_test, self.pca
    
    def get_processed_data(self):
        """
        Return preprocessed data.

        Returns:
            dict: preprocessed data and metadata
        """
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'n_features': self.X_train.shape[1] if self.X_train is not None else 0,
            'n_samples_train': self.X_train.shape[0] if self.X_train is not None else 0,
            'n_samples_test': self.X_test.shape[0] if self.X_test is not None else 0
        }
    
    def full_pipeline(self):
        """
        Run the full data preprocessing pipeline.

        Executes: load data -> handle missing values -> remove outliers -> separate features/labels
        -> normalize -> PCA

        Returns:
            dict: preprocessed data
        """
        logging.info("Starting full data preprocessing pipeline...")
        
        # 1. Data loading
        self.load_data()
        
        # 2. Check and handle missing values
        self.check_missing_values()
        self.handle_missing_values()
        
        # 3. Remove outliers
        self.remove_outliers()
        
        # 4. Separate features and labels
        self.separate_features_labels()
        
        # 5. Feature selection (run before normalization)
        self.select_features()
        
        # 6. Data normalization
        self.normalize_data()
        
        # 7. Apply PCA
        self.apply_pca()
        
        logging.info("Data preprocessing pipeline completed!")
        
        return self.get_processed_data()

# Functional convenience interface
def load_and_preprocess_data(config=None):
    """
    Convenience function to perform data loading and preprocessing in one call.

    Args:
        config (Config, optional): configuration object

    Returns:
        dict: preprocessed data
    """
    loader = DataLoader(config)
    return loader.full_pipeline()

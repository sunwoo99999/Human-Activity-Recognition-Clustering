"""
main.py
Main entry for the HAR dataset K-Means analysis program.

This script runs the full analysis pipeline in order:
1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. K-Means clustering
4. Clustering evaluation
5. Statistical significance testing
6. Visualization
7. Report generation

Usage:
    python main.py
"""

import sys
import logging
from datetime import datetime

# Project module imports
from config import Config
from utils import setup_seed, setup_logging, Timer, print_section_header
from data_loader import DataLoader
from eda import EDA
from clustering import KMeansClustering
from evaluation import ClusteringEvaluator
from statistical_tests import StatisticalTester
from visualization import create_all_visualizations
from report_generator import create_report


def main():
    """
    Main entry function

    Runs the full analysis pipeline in sequence.
    """
    # ========================================
    # 0. Initial setup
    # ========================================
    print_section_header("HAR Dataset K-Means Clustering Analysis", char='=', width=80)
    
    # Create result directories
    Config.create_directories()
    
    # Setup logging
    setup_logging()
    
    # Set random seed
    setup_seed()
    
    logging.info("="*80)
    logging.info("HAR K-Means Clustering Analysis Started")
    logging.info("="*80)
    
    # Measure total runtime
    total_timer = Timer("Total Analysis Pipeline")
    total_timer.__enter__()
    
    try:
        # ========================================
        # 1. Data loading and preprocessing
        # ========================================
        print_section_header("1. Data Loading and Preprocessing")
        
        data_loader = DataLoader()
        
        # 1-1. Data loading
        logging.info("Loading data...")
        data_loader.load_data()
        
        # 1-2. Check and handle missing values
        logging.info("Handling missing values...")
        data_loader.check_missing_values()
        data_loader.handle_missing_values()
        
        # 1-3. Remove outliers
        logging.info("Removing outliers...")
        data_loader.remove_outliers()
        
        # 1-4. Separate features and labels
        logging.info("Separating features and labels...")
        data_loader.separate_features_labels()
        
        # 1-5. Feature selection (optional)
        if Config.USE_FEATURE_SELECTION:
            logging.info("Selecting important features...")
            data_loader.select_features()
        
        # 1-6. Data normalization
        logging.info("Normalizing data...")
        data_loader.normalize_data()
        
        # 1-7. Apply PCA (optional)
        if Config.USE_PCA:
            logging.info("Applying PCA...")
            data_loader.apply_pca()
        
        # Retrieve preprocessed data
        data_dict = data_loader.get_processed_data()
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        
        # Update feature names when using PCA
        if Config.USE_PCA:
            n_components = X_train.shape[1]
            feature_names = [f"PC{i+1}" for i in range(n_components)]
            logging.info(f"Feature names updated for PCA: {len(feature_names)} components")
        
        logging.info(f"Data preprocessing completed!")
        logging.info(f"Final train shape: {X_train.shape}")
        logging.info(f"Final test shape: {X_test.shape}")
        
        # ========================================
        # 2. Exploratory Data Analysis (EDA)
        # ========================================
        print_section_header("2. Exploratory Data Analysis (EDA)")
        
        eda = EDA(X_train, y_train, X_test, y_test, feature_names)
        eda_results = eda.full_analysis()
        
        logging.info("EDA completed!")
        
        # ========================================
        # 3. K-Means Clustering
        # ========================================
        print_section_header("3. K-Means Clustering")
        
        clustering = KMeansClustering()
        
        # 3-1. Perform clustering for multiple K values
        logging.info(f"Performing K-Means for K range: {Config.K_RANGE}")
        clustering.fit_multiple_k(X_train)
        
        # 3-2. Find optimal K
        logging.info("Finding optimal K...")
        optimal_k = clustering.find_optimal_k(y_true=y_train)
        
        # 3-3. Cluster analysis
        logging.info("Analyzing clusters...")
        cluster_stats = clustering.analyze_clusters(X_train)
        
        # 3-4. Predict labels for train and test data
        train_labels = clustering.get_cluster_labels()
        test_labels = clustering.predict(X_test) if X_test is not None else None
        
        logging.info(f"Clustering completed! Optimal K = {optimal_k}")
        
        # ========================================
        # 4. Clustering Evaluation
        # ========================================
        print_section_header("4. Clustering Evaluation")
        
        # Evaluate on training data
        evaluator_train = ClusteringEvaluator(X_train, train_labels, y_train)
        eval_results = evaluator_train.full_evaluation()
        
        logging.info("Clustering evaluation completed!")
        
        # ========================================
        # 5. Statistical Significance Testing
        # ========================================
        print_section_header("5. Statistical Significance Testing")
        
        tester = StatisticalTester(X_train, train_labels, feature_names)
        stat_results = tester.full_statistical_analysis(y_train)
        
        # Reproducibility test (optional)
        if Config.N_ITERATIONS > 1:
            logging.info("Performing reproducibility test...")
            reproducibility_results = tester.repeated_experiments(
                X_train, 
                n_iterations=Config.N_ITERATIONS
            )
            stat_results['reproducibility'] = reproducibility_results
        
        logging.info("Statistical testing completed!")
        
        # ========================================
        # 6. Visualization
        # ========================================
        print_section_header("6. Visualization Generation")
        
        create_all_visualizations(
            clustering_obj=clustering,
            X_train=X_train,
            train_labels=train_labels,
            eda_results=eda_results,
            eval_results=eval_results,
            stat_results=stat_results
        )
        
        logging.info("Visualizations completed!")
        
        # ========================================
        # 7. Report generation
        # ========================================
        print_section_header("7. Report Generation")
        
        create_report(
            data_dict=data_dict,
            eda_results=eda_results,
            clustering_obj=clustering,
            eval_results=eval_results,
            stat_results=stat_results
        )
        
        logging.info("Report generation completed!")
        
        # ========================================
        # 8. Print analysis summary
        # ========================================
        print_section_header("8. Analysis Summary")
        
        print(f"\n{'='*80}")
        print(f"{'Analysis Complete!':^80}")
        print(f"{'='*80}\n")
        
        print(f"Data:")
        print(f"  - Training samples: {X_train.shape[0]:,}")
        print(f"  - Test samples: {X_test.shape[0]:,}")
        print(f"  - Number of features: {X_train.shape[1]}")
        
        print(f"\nClustering:")
        print(f"  - Selected K: {optimal_k}")
        print(f"  - Silhouette Score: {eval_results['internal_metrics']['silhouette_score']:.4f}")
        print(f"  - Davies-Bouldin Index: {eval_results['internal_metrics']['davies_bouldin_index']:.4f}")
        print(f"  - Calinski-Harabasz Index: {eval_results['internal_metrics']['calinski_harabasz_index']:.2f}")
        
        if 'external_metrics' in eval_results and eval_results['external_metrics']:
            print(f"\nExternal metrics:")
            print(f"  - Adjusted Rand Index: {eval_results['external_metrics']['adjusted_rand_index']:.4f}")
            print(f"  - Normalized Mutual Information: {eval_results['external_metrics']['normalized_mutual_info']:.4f}")
            print(f"  - Purity: {eval_results['external_metrics']['purity']:.4f}")
        
        print(f"\nStatistical testing:")
        if 'silhouette' in stat_results:
            print(f"  - Silhouette Score: {stat_results['silhouette']['silhouette_score']:.4f}")
        
        print(f"\nResults saved to:")
        print(f"  - Figures: {Config.FIGURES_DIR}")
        print(f"  - Tables: {Config.TABLES_DIR}")
        print(f"  - Report: {Config.get_report_path()}")
        print(f"  - Logs: {Config.get_log_path()}")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        logging.error(f"Error occurred during analysis: {str(e)}", exc_info=True)
        print(f"\n❌ Error occurred: {str(e)}")
        print(f"See log file for details: {Config.get_log_path()}")
        sys.exit(1)
    
    finally:
        # Print total runtime
        total_timer.__exit__(None, None, None)
        
        logging.info("="*80)
        logging.info("HAR K-Means Clustering Analysis Completed")
        logging.info("="*80)


if __name__ == "__main__":
    """
    Call `main()` when script is executed directly.
    """
    main()

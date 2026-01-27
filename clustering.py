"""
clustering.py
K-Means clustering module

Provides K-Means fitting, optimal-K selection, and cluster analysis utilities.
"""

import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from config import Config
from utils import Timer


class KMeansClustering:
    """
    Class that performs K-Means clustering and analyzes results.
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (Config, optional): configuration object
        """
        self.config = config if config is not None else Config()
        self.models = {}  # models keyed by K
        self.results = {}  # results keyed by K
        self.optimal_k = None
        self.optimal_model = None
    
    def fit_single_k(self, X, k, random_state=None):
        """
        Perform K-Means clustering for a specific K.

        Args:
            X (np.ndarray): input data
            k (int): number of clusters
            random_state (int, optional): random seed

        Returns:
            tuple: (model, labels, inertia)
        """
        if random_state is None:
            random_state = self.config.RANDOM_SEED
        
        # Create K-Means model
        kmeans = KMeans(
            n_clusters=k,
            init=self.config.KMEANS_INIT,
            max_iter=self.config.KMEANS_MAX_ITER,
            n_init=self.config.KMEANS_N_INIT,
            tol=self.config.KMEANS_TOL,
            random_state=random_state
        )
        
        # Fit and predict
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        
        return kmeans, labels, inertia
    
    def evaluate_clustering(self, X, labels, k):
        """
        Evaluate clustering results.

        Args:
            X (np.ndarray): input data
            labels (np.ndarray): cluster labels
            k (int): number of clusters

        Returns:
            dict: evaluation metrics
        """
        # Silhouette Score (higher is better, range -1..1)
        # Measures cohesion and separation
        silhouette = silhouette_score(X, labels)
        
        # Davies-Bouldin Index (lower is better)
        # Ratio of within-cluster to between-cluster distances
        davies_bouldin = davies_bouldin_score(X, labels)
        
        # Calinski-Harabasz Index (higher is better)
        # Ratio of between-cluster dispersion to within-cluster dispersion
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        return {
            'k': k,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz
        }
    
    def fit_multiple_k(self, X, k_range=None):
        """
        Run clustering and evaluation for multiple K values.

        Args:
            X (np.ndarray): input data
            k_range (list, optional): list of K values to try

        Returns:
            dict: results keyed by K
        """
        if k_range is None:
            k_range = self.config.K_RANGE
        
        with Timer(f"K-Means clustering for K={k_range}"):
            for k in k_range:
                logging.info(f"Fitting K-Means with K={k}...")

                # Perform clustering
                model, labels, inertia = self.fit_single_k(X, k)

                # Evaluation
                scores = self.evaluate_clustering(X, labels, k)
                scores['inertia'] = inertia
                scores['model'] = model
                scores['labels'] = labels

                # Save results
                self.models[k] = model
                self.results[k] = scores

                logging.info(f"  K={k}: Silhouette={scores['silhouette']:.4f}, "
                           f"Davies-Bouldin={scores['davies_bouldin']:.4f}, "
                           f"Inertia={inertia:.2f}")
        
        return self.results
    
    def find_optimal_k_elbow(self):
        """
        Find optimal K using the Elbow Method.

        Detect the "elbow" in the inertia (SSE) curve where the reduction
        rate of inertia sharply decreases.

        Returns:
            int: optimal K value
        """
        k_values = sorted(self.results.keys())
        inertias = [self.results[k]['inertia'] for k in k_values]
        
        # Compute inertia differences
        inertia_diffs = np.diff(inertias)
        
        # Compute second differences of inertia changes
        # Find the index where the change decreases most sharply (elbow)
        if len(inertia_diffs) > 1:
            second_diffs = np.diff(inertia_diffs)
            # Find index with largest second derivative (elbow)
            elbow_idx = np.argmax(second_diffs) + 1
            optimal_k = k_values[elbow_idx]
        else:
            # If only two K values are available, choose the middle
            optimal_k = k_values[len(k_values) // 2]
        
        logging.info(f"Optimal K by Elbow Method: {optimal_k}")
        return optimal_k
    
    def find_optimal_k_silhouette(self):
        """
        Find the K with the maximum Silhouette Score.

        Returns:
            int: optimal K value
        """
        k_values = sorted(self.results.keys())
        silhouette_scores = [self.results[k]['silhouette'] for k in k_values]
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = k_values[optimal_idx]
        
        logging.info(f"Optimal K by Silhouette Score: {optimal_k} "
                f"(score={silhouette_scores[optimal_idx]:.4f})")
        return optimal_k
    
    def find_optimal_k_davies_bouldin(self):
        """
        Find the K with the minimum Davies-Bouldin Index.

        Returns:
            int: optimal K value
        """
        k_values = sorted(self.results.keys())
        db_scores = [self.results[k]['davies_bouldin'] for k in k_values]
        
        optimal_idx = np.argmin(db_scores)
        optimal_k = k_values[optimal_idx]
        
        logging.info(f"Optimal K by Davies-Bouldin Index: {optimal_k} "
                f"(score={db_scores[optimal_idx]:.4f})")
        return optimal_k
    
    def find_optimal_k_calinski_harabasz(self):
        """
        Find the K with the maximum Calinski-Harabasz Index.

        Returns:
            int: optimal K value
        """
        k_values = sorted(self.results.keys())
        ch_scores = [self.results[k]['calinski_harabasz'] for k in k_values]
        
        optimal_idx = np.argmax(ch_scores)
        optimal_k = k_values[optimal_idx]
        
        logging.info(f"Optimal K by Calinski-Harabasz Index: {optimal_k} "
                f"(score={ch_scores[optimal_idx]:.2f})")
        return optimal_k
    
    def find_optimal_k_composite(self, y_true):
        """
        Find optimal K using a composite score

        Composite score is a weighted combination of Silhouette, (inverse) Davies-Bouldin, and Purity.

        Args:
            y_true (np.ndarray): ground-truth labels (required for purity calculation)

        Returns:
            int: optimal K value
        """
        k_values = sorted(self.results.keys())
        weights = self.config.COMPOSITE_WEIGHTS
        
        composite_scores = []
        
        for k in k_values:
            # Extract metrics for each K
            silhouette = self.results[k]['silhouette']
            davies_bouldin = self.results[k]['davies_bouldin']
            
            # Purity calculation without creating an evaluator
            labels = self.results[k]['labels']
            confusion_mat = np.zeros((len(np.unique(y_true)), k))
            
            for true_class_idx, true_class in enumerate(np.unique(y_true)):
                for cluster_idx in range(k):
                    confusion_mat[true_class_idx, cluster_idx] = np.sum(
                        (y_true == true_class) & (labels == cluster_idx)
                    )
            
            purity = np.sum(np.max(confusion_mat, axis=0)) / len(y_true)
            
            # Davies-Bouldin is better when lower, so invert it for normalization
            db_normalized = 1.0 / (1.0 + davies_bouldin)
            
            # Compute composite score (weighted average)
            composite = (weights['silhouette'] * silhouette + 
                         weights['davies_bouldin'] * db_normalized + 
                         weights['purity'] * purity)
            
            composite_scores.append(composite)
            
            logging.info(f"K={k}: Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}, "
                        f"Purity={purity:.4f}, Composite={composite:.4f}")
        
        optimal_idx = np.argmax(composite_scores)
        optimal_k = k_values[optimal_idx]

        logging.info(f"Optimal K by Composite Score: {optimal_k} "
                    f"(score={composite_scores[optimal_idx]:.4f})")
        return optimal_k
    
    def find_optimal_k(self, criterion=None, y_true=None):
        """
        Find optimal K using the specified criterion.

        Args:
            criterion (str, optional): one of 'silhouette', 'davies_bouldin',
                'calinski_harabasz', 'elbow', 'composite'
            y_true (np.ndarray, optional): ground-truth labels (required for composite)

        Returns:
            int: optimal K value
        """
        if criterion is None:
            criterion = self.config.OPTIMAL_K_CRITERION
        
        with Timer(f"Finding optimal K using {criterion}"):
            if criterion == 'silhouette':
                self.optimal_k = self.find_optimal_k_silhouette()
            elif criterion == 'davies_bouldin':
                self.optimal_k = self.find_optimal_k_davies_bouldin()
            elif criterion == 'calinski_harabasz':
                self.optimal_k = self.find_optimal_k_calinski_harabasz()
            elif criterion == 'elbow':
                self.optimal_k = self.find_optimal_k_elbow()
            elif criterion == 'composite':
                if y_true is None:
                    raise ValueError("y_true is required for composite criterion")
                self.optimal_k = self.find_optimal_k_composite(y_true)
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            # Save optimal model
            self.optimal_model = self.models[self.optimal_k]
            
            logging.info(f"Selected optimal K={self.optimal_k}")
        
        return self.optimal_k
    
    def get_cluster_centers(self, k=None):
        """
        Return cluster centers (centroids)

        Args:
            k (int, optional): K value. If None, uses the optimal K.

        Returns:
            np.ndarray: array of cluster centers
        """
        if k is None:
            k = self.optimal_k
        
        if k not in self.models:
            raise ValueError(f"Model for K={k} not found")
        
        return self.models[k].cluster_centers_
    
    def get_cluster_labels(self, k=None):
        """
        Return cluster labels

        Args:
            k (int, optional): K value. If None, uses the optimal K.

        Returns:
            np.ndarray: cluster label array
        """
        if k is None:
            k = self.optimal_k
        
        if k not in self.results:
            raise ValueError(f"Results for K={k} not found")
        
        return self.results[k]['labels']
    
    def predict(self, X, k=None):
        """
        Predict cluster assignments for new data

        Args:
            X (np.ndarray): input data
            k (int, optional): K value. If None, uses the optimal K.

        Returns:
            np.ndarray: predicted cluster labels
        """
        if k is None:
            k = self.optimal_k
        
        if k not in self.models:
            raise ValueError(f"Model for K={k} not found")
        
        return self.models[k].predict(X)
    
    def analyze_clusters(self, X, k=None):
        """
        Per-cluster statistical analysis

        Args:
            X (np.ndarray): input data
            k (int, optional): K value. If None, uses the optimal K.

        Returns:
            dict: statistics per cluster
        """
        if k is None:
            k = self.optimal_k
        
        labels = self.get_cluster_labels(k)
        centers = self.get_cluster_centers(k)
        
        cluster_stats = {}
        for cluster_id in range(k):
            # Samples belonging to the current cluster
            cluster_mask = labels == cluster_id
            cluster_samples = X[cluster_mask]

            # Compute statistics
            cluster_stats[cluster_id] = {
                'size': cluster_samples.shape[0],
                'percentage': (cluster_samples.shape[0] / X.shape[0]) * 100,
                'center': centers[cluster_id],
                'mean': cluster_samples.mean(axis=0),
                'std': cluster_samples.std(axis=0),
                'min': cluster_samples.min(axis=0),
                'max': cluster_samples.max(axis=0)
            }
            
            logging.info(f"Cluster {cluster_id}: {cluster_stats[cluster_id]['size']} samples "
                        f"({cluster_stats[cluster_id]['percentage']:.2f}%)")
        
        return cluster_stats
    
    def get_summary(self):
        """
        Return a summary of overall clustering results.

        Returns:
            dict: summary information
        """
        if not self.results:
            logging.warning("No clustering results available")
            return {}
        
        summary = {
            'k_range': list(self.results.keys()),
            'optimal_k': self.optimal_k,
            'optimal_scores': self.results[self.optimal_k] if self.optimal_k else None,
            'all_results': self.results
        }
        
        return summary


def perform_clustering(X_train, X_test=None, config=None):
    """
    Convenience function to run K-Means clustering.

    Args:
        X_train (np.ndarray): training data
        X_test (np.ndarray, optional): test data
        config (Config, optional): configuration object

    Returns:
        tuple: (clustering_obj, train_labels, test_labels)
    """
    clustering = KMeansClustering(config)
    
    # Run clustering for multiple K values
    clustering.fit_multiple_k(X_train)
    
    # Find optimal K
    clustering.find_optimal_k()
    
    # Training data labels
    train_labels = clustering.get_cluster_labels()
    
    # Test data labels (if available)
    test_labels = None
    if X_test is not None:
        test_labels = clustering.predict(X_test)
    
    # Cluster analysis
    clustering.analyze_clusters(X_train)
    
    return clustering, train_labels, test_labels

"""
evaluation.py
클러스터링 결과 평가 모듈

내부 평가 지표(레이블 불필요)와 외부 평가 지표(레이블 필요)를 계산합니다.
"""

import numpy as np
import logging
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    confusion_matrix
)
from config import Config
from utils import Timer


class ClusteringEvaluator:
    """
    클러스터링 결과를 다양한 지표로 평가하는 클래스
    """
    
    def __init__(self, X, cluster_labels, true_labels=None):
        """
        Args:
            X (np.ndarray): 입력 데이터
            cluster_labels (np.ndarray): 클러스터링 결과 레이블
            true_labels (np.ndarray, optional): 실제 레이블 (외부 평가용)
        """
        self.X = X
        self.cluster_labels = cluster_labels
        self.true_labels = true_labels
        self.n_clusters = len(np.unique(cluster_labels))
    
    def internal_metrics(self):
        """
        내부 평가 지표 계산 (레이블 불필요)
        
        클러스터의 응집도와 분리도를 측정합니다.
        
        Returns:
            dict: 내부 평가 지표들
        """
        with Timer("Computing internal evaluation metrics"):
            metrics = {}
            
            # Silhouette Score: -1 ~ 1, 높을수록 좋음
            # 같은 클러스터 내 유사도와 다른 클러스터와의 차이를 측정
            metrics['silhouette_score'] = silhouette_score(self.X, self.cluster_labels)
            
            # Davies-Bouldin Index: 0 이상, 낮을수록 좋음
            # 클러스터 내 분산과 클러스터 간 거리의 비율
            metrics['davies_bouldin_index'] = davies_bouldin_score(self.X, self.cluster_labels)
            
            # Calinski-Harabasz Index: 0 이상, 높을수록 좋음
            # 클러스터 간 분산과 클러스터 내 분산의 비율 (F-통계량과 유사)
            metrics['calinski_harabasz_index'] = calinski_harabasz_score(self.X, self.cluster_labels)
            
            # Inertia (WCSS - Within-Cluster Sum of Squares)
            # 각 샘플과 클러스터 중심 간 거리의 제곱합
            metrics['inertia'] = self._calculate_inertia()
            
            logging.info("Internal metrics computed:")
            logging.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            logging.info(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
            logging.info(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")
            logging.info(f"  Inertia: {metrics['inertia']:.2f}")
        
        return metrics
    
    def _calculate_inertia(self):
        """
        Inertia (WCSS) 계산
        
        Returns:
            float: Inertia 값
        """
        inertia = 0.0
        for cluster_id in range(self.n_clusters):
            # 클러스터에 속한 샘플들
            cluster_mask = self.cluster_labels == cluster_id
            cluster_samples = self.X[cluster_mask]
            
            # 클러스터 중심
            cluster_center = cluster_samples.mean(axis=0)
            
            # 중심까지 거리의 제곱합
            distances = np.sum((cluster_samples - cluster_center) ** 2, axis=1)
            inertia += distances.sum()
        
        return inertia
    
    def external_metrics(self):
        """
        외부 평가 지표 계산 (실제 레이블 필요)
        
        클러스터링 결과와 실제 레이블 간의 일치도를 측정합니다.
        
        Returns:
            dict: 외부 평가 지표들
        """
        if self.true_labels is None:
            logging.warning("True labels not available for external metrics")
            return {}
        
        with Timer("Computing external evaluation metrics"):
            metrics = {}
            
            # Adjusted Rand Index (ARI): -1 ~ 1, 높을수록 좋음, 1이면 완벽한 일치
            # 무작위 레이블링에 대해 보정된 Rand Index
            metrics['adjusted_rand_index'] = adjusted_rand_score(
                self.true_labels, self.cluster_labels
            )
            
            # Normalized Mutual Information (NMI): 0 ~ 1, 높을수록 좋음
            # 클러스터와 실제 클래스 간 상호정보량
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                self.true_labels, self.cluster_labels
            )
            
            # Homogeneity, Completeness, V-measure
            # Homogeneity: 각 클러스터가 단일 클래스만 포함하는 정도
            # Completeness: 같은 클래스가 같은 클러스터에 할당된 정도
            # V-measure: Homogeneity와 Completeness의 조화 평균
            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
                self.true_labels, self.cluster_labels
            )
            metrics['homogeneity'] = homogeneity
            metrics['completeness'] = completeness
            metrics['v_measure'] = v_measure
            
            # Purity: 각 클러스터에서 가장 많은 클래스의 비율
            metrics['purity'] = self._calculate_purity()
            
            logging.info("External metrics computed:")
            logging.info(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
            logging.info(f"  Normalized Mutual Information: {metrics['normalized_mutual_info']:.4f}")
            logging.info(f"  Homogeneity: {metrics['homogeneity']:.4f}")
            logging.info(f"  Completeness: {metrics['completeness']:.4f}")
            logging.info(f"  V-measure: {metrics['v_measure']:.4f}")
            logging.info(f"  Purity: {metrics['purity']:.4f}")
        
        return metrics
    
    def _calculate_purity(self):
        """
        Purity 계산
        
        각 클러스터에서 가장 빈번한 클래스의 비율의 가중 평균
        
        Returns:
            float: Purity 값 (0 ~ 1)
        """
        purity = 0.0
        n_samples = len(self.cluster_labels)
        
        for cluster_id in range(self.n_clusters):
            # 클러스터에 속한 샘플들의 실제 레이블
            cluster_mask = self.cluster_labels == cluster_id
            cluster_true_labels = self.true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                # 가장 빈번한 클래스의 개수
                unique, counts = np.unique(cluster_true_labels, return_counts=True)
                max_count = counts.max()
                purity += max_count
        
        purity /= n_samples
        return purity
    
    def silhouette_analysis(self):
        """
        샘플별 Silhouette Score 계산
        
        각 샘플의 silhouette coefficient를 계산하여
        클러스터별 품질을 상세히 분석합니다.
        
        Returns:
            dict: 클러스터별 silhouette 정보
        """
        with Timer("Computing silhouette analysis"):
            # 샘플별 silhouette score
            sample_silhouette_values = silhouette_samples(self.X, self.cluster_labels)
            
            # 클러스터별 평균 silhouette score
            cluster_silhouettes = {}
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.cluster_labels == cluster_id
                cluster_scores = sample_silhouette_values[cluster_mask]
                
                cluster_silhouettes[cluster_id] = {
                    'mean': cluster_scores.mean(),
                    'std': cluster_scores.std(),
                    'min': cluster_scores.min(),
                    'max': cluster_scores.max(),
                    'size': cluster_scores.shape[0]
                }
            
            # 전체 평균
            overall_silhouette = sample_silhouette_values.mean()
            
            logging.info(f"Overall Silhouette Score: {overall_silhouette:.4f}")
            for cluster_id, stats in cluster_silhouettes.items():
                logging.info(f"  Cluster {cluster_id}: mean={stats['mean']:.4f}, "
                           f"std={stats['std']:.4f}, size={stats['size']}")
            
            return {
                'overall': overall_silhouette,
                'per_cluster': cluster_silhouettes,
                'per_sample': sample_silhouette_values
            }
    
    def confusion_matrix_clustering(self):
        """
        클러스터링 결과에 대한 혼동 행렬 생성
        
        실제 레이블이 있을 때, 클러스터와 클래스 간의 매핑을 확인합니다.
        
        Returns:
            np.ndarray: 혼동 행렬
        """
        if self.true_labels is None:
            logging.warning("True labels not available for confusion matrix")
            return None
        
        with Timer("Computing confusion matrix"):
            # Convert cluster labels to strings to match true_labels type
            cluster_labels_str = [f"Cluster_{label}" for label in self.cluster_labels]
            conf_matrix = confusion_matrix(self.true_labels, cluster_labels_str)
            
            logging.info(f"Confusion matrix shape: {conf_matrix.shape}")
            logging.info(f"\n{conf_matrix}")
        
        return conf_matrix
    
    def cluster_distribution(self):
        """
        클러스터별 샘플 분포 계산
        
        Returns:
            dict: 클러스터별 샘플 수와 비율
        """
        distribution = {}
        n_samples = len(self.cluster_labels)
        
        unique_clusters, counts = np.unique(self.cluster_labels, return_counts=True)
        
        for cluster_id, count in zip(unique_clusters, counts):
            distribution[int(cluster_id)] = {
                'count': int(count),
                'percentage': (count / n_samples) * 100
            }
        
        logging.info("Cluster distribution:")
        for cluster_id, info in distribution.items():
            logging.info(f"  Cluster {cluster_id}: {info['count']} samples "
                        f"({info['percentage']:.2f}%)")
        
        return distribution
    
    def full_evaluation(self):
        """
        전체 평가 파이프라인 실행
        
        Returns:
            dict: 모든 평가 결과
        """
        logging.info("Starting full evaluation pipeline...")
        
        results = {}
        
        # 1. 내부 평가 지표
        results['internal_metrics'] = self.internal_metrics()
        
        # 2. 외부 평가 지표 (레이블이 있는 경우)
        if self.true_labels is not None:
            results['external_metrics'] = self.external_metrics()
            results['confusion_matrix'] = self.confusion_matrix_clustering()
        
        # 3. Silhouette 분석
        results['silhouette_analysis'] = self.silhouette_analysis()
        
        # 4. 클러스터 분포
        results['cluster_distribution'] = self.cluster_distribution()
        
        logging.info("Evaluation pipeline completed!")
        
        return results


def evaluate_clustering(X, cluster_labels, true_labels=None):
    """
    클러스터링 결과 평가 (편의 함수)
    
    Args:
        X (np.ndarray): 입력 데이터
        cluster_labels (np.ndarray): 클러스터링 결과 레이블
        true_labels (np.ndarray, optional): 실제 레이블
    
    Returns:
        dict: 평가 결과
    """
    evaluator = ClusteringEvaluator(X, cluster_labels, true_labels)
    return evaluator.full_evaluation()

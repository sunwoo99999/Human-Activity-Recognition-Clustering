"""
statistical_tests.py
통계적 유의성 검증 모듈

클러스터 간 차이 검정과 재현성 검증을 수행합니다.
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.stats import f_oneway
from config import Config
from utils import Timer


class StatisticalTester:
    """
    클러스터링 결과에 대한 통계적 검증을 수행하는 클래스
    """
    
    def __init__(self, X, cluster_labels, feature_names=None, alpha=0.05):
        """
        Args:
            X (np.ndarray): 입력 데이터
            cluster_labels (np.ndarray): 클러스터 레이블
            feature_names (list, optional): 특성 이름 리스트
            alpha (float): 유의수준 (기본값: 0.05)
        """
        self.X = X
        self.cluster_labels = cluster_labels
        self.feature_names = feature_names
        self.alpha = alpha
        self.n_clusters = len(np.unique(cluster_labels))
        
        # 특성 이름이 없으면 자동 생성
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    def anova_test(self):
        """
        일원 분산분석(One-way ANOVA) 수행
        
        각 특성에 대해 클러스터 간 평균 차이가 통계적으로 유의한지 검정합니다.
        
        Returns:
            pd.DataFrame: 각 특성별 ANOVA 결과 (F-statistic, p-value, significant)
        """
        with Timer("Performing ANOVA tests"):
            results = []
            
            for i, feature_name in enumerate(self.feature_names):
                # 특성 데이터
                feature_data = self.X[:, i]
                
                # 클러스터별로 데이터 분리
                groups = [
                    feature_data[self.cluster_labels == cluster_id]
                    for cluster_id in range(self.n_clusters)
                ]
                
                # ANOVA F-검정
                f_statistic, p_value = f_oneway(*groups)
                
                results.append({
                    'feature': feature_name,
                    'F_statistic': f_statistic,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                })
            
            results_df = pd.DataFrame(results)
            
            # 유의한 특성 개수
            n_significant = results_df['significant'].sum()
            logging.info(f"ANOVA: {n_significant}/{len(self.feature_names)} features "
                        f"show significant differences (alpha={self.alpha})")
            
            return results_df
    def repeated_experiments(self, X, n_iterations=10):
        """
        반복 실험을 통한 재현성 검증
        
        여러 번 클러스터링을 수행하여 결과의 안정성을 평가합니다.
        
        Args:
            X (np.ndarray): 입력 데이터
            n_iterations (int): 반복 횟수 (기본값: 10)
        
        Returns:
            dict: 반복 실험 결과 통계 (평균, 표준편차, 최소, 최대)
        """
        from clustering import KMeansClustering
        
        with Timer(f"Performing {n_iterations} repeated experiments"):
            silhouette_scores = []
            davies_bouldin_scores = []
            
            for i in range(n_iterations):
                # 매번 다른 random seed로 클러스터링
                clustering = KMeansClustering()
                _, labels, _ = clustering.fit_single_k(
                    X,
                    k=self.n_clusters,
                    random_state=Config.RANDOM_SEED + i
                )
                
                # 평가 지표 계산
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                sil_score = silhouette_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
                
                silhouette_scores.append(sil_score)
                davies_bouldin_scores.append(db_score)
            
            # 통계량 계산
            results = {
                'silhouette': {
                    'mean': np.mean(silhouette_scores),
                    'std': np.std(silhouette_scores),
                    'min': np.min(silhouette_scores),
                    'max': np.max(silhouette_scores),
                },
                'davies_bouldin': {
                    'mean': np.mean(davies_bouldin_scores),
                    'std': np.std(davies_bouldin_scores),
                    'min': np.min(davies_bouldin_scores),
                    'max': np.max(davies_bouldin_scores),
                }
            }
            
            logging.info(f"Repeated experiments results (n={n_iterations}):")
            logging.info(f"  Silhouette: {results['silhouette']['mean']:.4f} ± "
                        f"{results['silhouette']['std']:.4f}")
            logging.info(f"  Davies-Bouldin: {results['davies_bouldin']['mean']:.4f} ± "
                        f"{results['davies_bouldin']['std']:.4f}")
            
            return results
    
    def full_statistical_analysis(self, true_labels=None):
        """
        전체 통계 검정 파이프라인 실행
        
        Args:
            true_labels (np.ndarray, optional): 실제 레이블 (참고용)
        
        Returns:
            dict: 모든 통계 검정 결과
        """
        logging.info("Starting simplified statistical analysis...")
        
        results = {}
        
        # 1. ANOVA 검정 - 클러스터 간 차이의 통계적 유의성
        results['anova'] = self.anova_test()
        
        logging.info("Statistical analysis completed!")
        
        return results


def perform_statistical_tests(X, cluster_labels, true_labels=None, 
                             feature_names=None, alpha=0.05):
    """
    통계적 검정 수행 (편의 함수)
    
    Args:
        X (np.ndarray): 입력 데이터
        cluster_labels (np.ndarray): 클러스터 레이블
        true_labels (np.ndarray, optional): 실제 레이블
        feature_names (list, optional): 특성 이름
        alpha (float): 유의수준
    
    Returns:
        dict: 검정 결과
    """
    tester = StatisticalTester(X, cluster_labels, feature_names, alpha)
    return tester.full_statistical_analysis(true_labels)

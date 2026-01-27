"""
eda.py
탐색적 데이터 분석(Exploratory Data Analysis) 모듈

데이터의 기본 통계량, 분포, 상관관계 등을 분석하는 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from config import Config
from utils import Timer


class EDA:
    """
    탐색적 데이터 분석을 수행하는 클래스
    """
    
    def __init__(self, X_train, y_train=None, X_test=None, y_test=None, feature_names=None):
        """
        Args:
            X_train (np.ndarray): 학습 특성 데이터
            y_train (np.ndarray, optional): 학습 레이블 데이터
            X_test (np.ndarray, optional): 테스트 특성 데이터
            y_test (np.ndarray, optional): 테스트 레이블 데이터
            feature_names (list, optional): 특성 이름 리스트
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        
        # 특성 이름이 없으면 자동 생성
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    def basic_statistics(self):
        """
        기본 통계량 계산 (평균, 표준편차, 최소/최대값, 사분위수 등)
        
        Returns:
            pd.DataFrame: 통계량 데이터프레임
        """
        with Timer("Computing basic statistics"):
            # 데이터프레임으로 변환
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            
            # 기본 통계량 계산
            stats_df = df_train.describe()
            
            # 추가 통계량 계산
            stats_df.loc['variance'] = df_train.var()
            stats_df.loc['skewness'] = df_train.skew()
            stats_df.loc['kurtosis'] = df_train.kurtosis()
            
            logging.info("Basic statistics computed")
            logging.info(f"\nStatistics summary:\n{stats_df.head(10)}")
            
            return stats_df
    
    def correlation_analysis(self, method='pearson'):
        """
        특성 간 상관관계 분석
        
        Args:
            method (str): 상관계수 계산 방법 ('pearson', 'spearman', 'kendall')
        
        Returns:
            pd.DataFrame: 상관행렬
        """
        with Timer(f"Computing correlation matrix ({method})"):
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            
            # 상관행렬 계산
            corr_matrix = df_train.corr(method=method)
            
            # 높은 상관관계를 가진 특성 쌍 찾기 (|r| > 0.8)
            high_corr_pairs = []
            n_features = len(self.feature_names)
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append({
                            'feature1': self.feature_names[i],
                            'feature2': self.feature_names[j],
                            'correlation': corr_value
                        })
            
            if high_corr_pairs:
                logging.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
            else:
                logging.info("No highly correlated feature pairs found (threshold: 0.8)")
            
            return corr_matrix
    
    def class_distribution(self):
        """
        클래스(레이블) 분포 분석
        
        Returns:
            dict: 각 클래스의 샘플 수와 비율
        """
        if self.y_train is None:
            logging.warning("No labels available for class distribution analysis")
            return None
        
        with Timer("Analyzing class distribution"):
            unique_classes, counts = np.unique(self.y_train, return_counts=True)
            total_samples = len(self.y_train)
            
            distribution = {}
            for cls, count in zip(unique_classes, counts):
                distribution[cls] = {
                    'count': count,
                    'percentage': (count / total_samples) * 100
                }
            
            logging.info("Class distribution:")
            for cls, info in distribution.items():
                logging.info(f"  Class {cls}: {info['count']} samples ({info['percentage']:.2f}%)")
            
            # 불균형 확인 (가장 많은 클래스와 가장 적은 클래스의 비율)
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 2:
                logging.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            
            return distribution
    
    def feature_importance_analysis(self):
        """
        특성의 분산을 기반으로 중요도 추정
        
        Returns:
            pd.DataFrame: 특성별 분산 및 중요도
        """
        with Timer("Analyzing feature importance"):
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            
            # 각 특성의 분산 계산
            variances = df_train.var()
            
            # 분산 기준 정규화 (0-100 스케일)
            importance = (variances / variances.sum()) * 100
            
            # 데이터프레임으로 정리
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'variance': variances.values,
                'importance': importance.values
            }).sort_values('importance', ascending=False)
            
            logging.info("Top 10 most important features by variance:")
            logging.info(f"\n{importance_df.head(10)}")
            
            return importance_df
    
    def normality_test(self, n_features=5):
        """
        주요 특성에 대한 정규성 검정 (Shapiro-Wilk test)
        
        Args:
            n_features (int): 검정할 특성 수 (분산 기준 상위)
        
        Returns:
            list: 각 특성의 검정 결과
        """
        with Timer("Performing normality tests"):
            # 분산 기준으로 상위 특성 선택
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            variances = df_train.var().sort_values(ascending=False)
            top_features = variances.head(n_features).index.tolist()
            
            results = []
            for feature in top_features:
                data = df_train[feature].values
                
                # 샘플 크기가 너무 크면 샘플링 (Shapiro-Wilk는 5000개까지만 권장)
                if len(data) > 5000:
                    data = np.random.choice(data, 5000, replace=False)
                
                # Shapiro-Wilk 검정
                statistic, p_value = stats.shapiro(data)
                is_normal = p_value > 0.05
                
                results.append({
                    'feature': feature,
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': is_normal
                })
                
                logging.info(f"  {feature}: p-value={p_value:.4f}, normal={is_normal}")
            
            return results
    
    def detect_outliers_zscore(self, threshold=3):
        """
        Z-score 방법으로 이상치 비율 계산
        
        Args:
            threshold (float): Z-score 임계값 (기본값: 3)
        
        Returns:
            dict: 각 특성별 이상치 비율
        """
        with Timer("Detecting outliers using Z-score"):
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            
            outlier_ratios = {}
            for feature in self.feature_names:
                data = df_train[feature].values
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > threshold
                outlier_ratio = (outliers.sum() / len(data)) * 100
                outlier_ratios[feature] = outlier_ratio
            
            # 이상치 비율이 높은 특성 출력
            high_outlier_features = {k: v for k, v in outlier_ratios.items() if v > 5}
            if high_outlier_features:
                logging.info(f"Features with >5% outliers: {len(high_outlier_features)}")
            
            return outlier_ratios
    
    def pairwise_feature_analysis(self, n_features=5):
        """
        주요 특성 간의 통계적 관계 분석
        
        Args:
            n_features (int): 분석할 특성 수
        
        Returns:
            pd.DataFrame: 특성 쌍별 통계 정보
        """
        with Timer("Pairwise feature analysis"):
            # 분산 기준으로 상위 특성 선택
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            variances = df_train.var().sort_values(ascending=False)
            top_features = variances.head(n_features).index.tolist()
            
            pairwise_stats = []
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    # Pearson 상관계수
                    corr, p_value = stats.pearsonr(df_train[feat1], df_train[feat2])
                    
                    pairwise_stats.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
            
            stats_df = pd.DataFrame(pairwise_stats)
            logging.info(f"Analyzed {len(pairwise_stats)} feature pairs")
            
            return stats_df
    
    def full_analysis(self):
        """
        전체 EDA 파이프라인 실행
        
        Returns:
            dict: 모든 분석 결과
        """
        logging.info("Starting full EDA pipeline...")
        
        results = {}
        
        # 1. 기본 통계량
        results['basic_stats'] = self.basic_statistics()
        
        # 2. 상관관계 분석
        results['correlation'] = self.correlation_analysis()
        
        # 3. 클래스 분포 (레이블이 있는 경우)
        if self.y_train is not None:
            results['class_distribution'] = self.class_distribution()
        
        # 4. 특성 중요도
        results['feature_importance'] = self.feature_importance_analysis()
        
        # 5. 정규성 검정 (상위 5개 특성)
        results['normality_tests'] = self.normality_test(n_features=5)
        
        # 6. 이상치 탐지
        results['outlier_ratios'] = self.detect_outliers_zscore()
        
        # 7. 특성 쌍별 분석
        results['pairwise_analysis'] = self.pairwise_feature_analysis(n_features=5)
        
        logging.info("EDA pipeline completed!")
        
        return results


def perform_eda(data_dict):
    """
    전처리된 데이터에 대한 EDA 수행 (편의 함수)
    
    Args:
        data_dict (dict): DataLoader에서 반환된 데이터 딕셔너리
    
    Returns:
        dict: EDA 결과
    """
    eda = EDA(
        X_train=data_dict['X_train'],
        y_train=data_dict.get('y_train'),
        X_test=data_dict.get('X_test'),
        y_test=data_dict.get('y_test'),
        feature_names=data_dict.get('feature_names')
    )
    
    return eda.full_analysis()

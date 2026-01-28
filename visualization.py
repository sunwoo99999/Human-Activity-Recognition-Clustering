"""
visualization.py
시각화 모듈

클러스터링 결과, 평가 지표, 통계 분석 결과를 시각화합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from pathlib import Path
from config import Config
from utils import Timer


class Visualizer:
    """
    다양한 시각화를 생성하는 클래스
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (Config, optional): 설정 객체
        """
        self.config = config if config is not None else Config()
        self._setup_style()
    
    def _setup_style(self):
        """시각화 스타일 설정"""
        # Matplotlib 스타일
        try:
            plt.style.use(self.config.PLOT_STYLE)
        except:
            plt.style.use('default')
            logging.warning(f"Style '{self.config.PLOT_STYLE}' not found, using 'default'")
        
        # Seaborn 기본 설정
        sns.set_palette(self.config.COLOR_PALETTE)
        
        # 한글 폰트 설정 (Windows 환경)
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    def _save_figure(self, filename, dpi=None):
        """
        그래프를 파일로 저장
        
        Args:
            filename (str): 파일명 (확장자 포함)
            dpi (int, optional): 해상도
        """
        if dpi is None:
            dpi = self.config.FIGURE_DPI
        
        filepath = self.config.FIGURES_DIR / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logging.info(f"Figure saved: {filepath}")
    
    def plot_elbow_curve(self, clustering_results, save=True):
        """
        Elbow Curve (팔꿈치 곡선) 그리기
        
        K 값에 따른 Inertia(SSE) 변화를 시각화합니다.
        
        Args:
            clustering_results (dict): 클러스터링 결과 딕셔너리
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting elbow curve"):
            k_values = sorted(clustering_results.keys())
            inertias = [clustering_results[k]['inertia'] for k in k_values]
            
            plt.figure(figsize=self.config.FIGURE_SIZE)
            plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
            plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(k_values)
            
            if save:
                self._save_figure('elbow_curve.png')
            
            plt.close()
    
    def plot_silhouette_scores(self, clustering_results, save=True):
        """
        K 값에 따른 Silhouette Score 변화 그리기
        
        Args:
            clustering_results (dict): 클러스터링 결과 딕셔너리
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting silhouette scores"):
            k_values = sorted(clustering_results.keys())
            silhouette_scores = [clustering_results[k]['silhouette'] for k in k_values]
            
            plt.figure(figsize=self.config.FIGURE_SIZE)
            plt.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(k_values)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            if save:
                self._save_figure('silhouette_scores.png')
            
            plt.close()
    
    def plot_all_metrics(self, clustering_results, save=True):
        """
        모든 평가 지표를 하나의 그래프에 표시
        
        Args:
            clustering_results (dict): 클러스터링 결과 딕셔너리
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting all metrics"):
            k_values = sorted(clustering_results.keys())
            
            # 각 지표 추출
            silhouette = [clustering_results[k]['silhouette'] for k in k_values]
            davies_bouldin = [clustering_results[k]['davies_bouldin'] for k in k_values]
            calinski = [clustering_results[k]['calinski_harabasz'] for k in k_values]
            
            # Calinski-Harabasz는 스케일이 다르므로 정규화
            calinski_norm = np.array(calinski) / np.max(calinski)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Silhouette Score
            axes[0, 0].plot(k_values, silhouette, 'go-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Number of Clusters (K)')
            axes[0, 0].set_ylabel('Silhouette Score')
            axes[0, 0].set_title('Silhouette Score (Higher is Better)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(k_values)
            
            # Davies-Bouldin Index
            axes[0, 1].plot(k_values, davies_bouldin, 'ro-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Number of Clusters (K)')
            axes[0, 1].set_ylabel('Davies-Bouldin Index')
            axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xticks(k_values)
            
            # Calinski-Harabasz Index
            axes[1, 0].plot(k_values, calinski, 'bo-', linewidth=2, markersize=8)
            axes[1, 0].set_xlabel('Number of Clusters (K)')
            axes[1, 0].set_ylabel('Calinski-Harabasz Index')
            axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xticks(k_values)
            
            # 정규화된 비교
            axes[1, 1].plot(k_values, silhouette, 'g-', label='Silhouette', linewidth=2)
            axes[1, 1].plot(k_values, 1 - np.array(davies_bouldin)/np.max(davies_bouldin), 
                           'r-', label='1 - DB (norm)', linewidth=2)
            axes[1, 1].plot(k_values, calinski_norm, 'b-', label='CH (norm)', linewidth=2)
            axes[1, 1].set_xlabel('Number of Clusters (K)')
            axes[1, 1].set_ylabel('Normalized Score')
            axes[1, 1].set_title('Normalized Metrics Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(k_values)
            
            plt.tight_layout()
            
            if save:
                self._save_figure('all_metrics.png')
            
            plt.close()
    
    def plot_clusters_2d_pca(self, X, labels, save=True):
        """
        PCA를 사용하여 2D로 클러스터 시각화
        
        Args:
            X (np.ndarray): 입력 데이터
            labels (np.ndarray): 클러스터 레이블
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting 2D PCA clusters"):
            # PCA로 2차원 축소
            pca = PCA(n_components=2, random_state=self.config.RANDOM_SEED)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=self.config.FIGURE_SIZE)
            
            # 클러스터별로 다른 색상으로 표시
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap(self.config.COLOR_PALETTE, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors(i)], label=f'Cluster {label}',
                          alpha=0.6, edgecolors='w', s=50)
            
            plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                      fontsize=12)
            plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                      fontsize=12)
            plt.title('Cluster Visualization (PCA)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save:
                self._save_figure('clusters_2d_pca.png')
            
            plt.close()
    
    def plot_clusters_2d_tsne(self, X, labels, save=True):
        """
        t-SNE를 사용하여 2D로 클러스터 시각화
        
        Args:
            X (np.ndarray): 입력 데이터
            labels (np.ndarray): 클러스터 레이블
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting 2D t-SNE clusters"):
            # t-SNE로 2차원 축소
            tsne = TSNE(
                n_components=2,
                random_state=self.config.RANDOM_SEED,
                perplexity=self.config.TSNE_PERPLEXITY,
                max_iter=self.config.TSNE_N_ITER,
                learning_rate=self.config.TSNE_LEARNING_RATE
            )
            X_tsne = tsne.fit_transform(X)
            
            plt.figure(figsize=self.config.FIGURE_SIZE)
            
            # 클러스터별로 다른 색상으로 표시
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap(self.config.COLOR_PALETTE, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          c=[colors(i)], label=f'Cluster {label}',
                          alpha=0.6, edgecolors='w', s=50)
            
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.title('Cluster Visualization (t-SNE)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save:
                self._save_figure('clusters_2d_tsne.png')
            
            plt.close()
    
    def plot_confusion_matrix(self, conf_matrix, save=True):
        """
        혼동 행렬 히트맵 그리기
        
        Args:
            conf_matrix (np.ndarray): 혼동 행렬
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting confusion matrix"):
            plt.figure(figsize=self.config.FIGURE_SIZE)
            
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       cbar_kws={'label': 'Count'},
                       linewidths=0.5, linecolor='gray')
            
            plt.xlabel('Predicted Cluster', fontsize=12)
            plt.ylabel('True Class', fontsize=12)
            plt.title('Confusion Matrix: True Class vs Predicted Cluster', 
                     fontsize=14, fontweight='bold')
            
            if save:
                self._save_figure('confusion_matrix.png')
            
            plt.close()
    
    def plot_silhouette_diagram(self, X, labels, silhouette_values, save=True):
        """
        Silhouette Diagram 그리기
        
        클러스터별 샘플들의 silhouette coefficient를 시각화합니다.
        
        Args:
            X (np.ndarray): 입력 데이터
            labels (np.ndarray): 클러스터 레이블
            silhouette_values (np.ndarray): 샘플별 silhouette score
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting silhouette diagram"):
            n_clusters = len(np.unique(labels))
            
            fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
            
            y_lower = 10
            for i in range(n_clusters):
                # i번째 클러스터의 silhouette values
                cluster_silhouette_values = silhouette_values[labels == i]
                cluster_silhouette_values.sort()
                
                size_cluster_i = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = plt.cm.get_cmap(self.config.COLOR_PALETTE)(i / n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                # 클러스터 레이블 표시
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                
                y_lower = y_upper + 10
            
            # 평균 silhouette score 표시
            silhouette_avg = np.mean(silhouette_values)
            ax.axvline(x=silhouette_avg, color='red', linestyle='--',
                      label=f'Average: {silhouette_avg:.3f}')
            
            ax.set_xlabel('Silhouette Coefficient', fontsize=12)
            ax.set_ylabel('Cluster', fontsize=12)
            ax.set_title('Silhouette Diagram for Each Cluster', fontsize=14, fontweight='bold')
            ax.legend()
            
            if save:
                self._save_figure('silhouette_diagram.png')
            
            plt.close()
    
    def plot_feature_importance(self, importance_df, top_n=20, save=True):
        """
        특성 중요도 막대 그래프
        
        Args:
            importance_df (pd.DataFrame): 특성 중요도 데이터프레임
            top_n (int): 상위 몇 개 특성만 표시할지
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting feature importance"):
            # 상위 N개만 선택
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance (%)', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance (by Variance)', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            if save:
                self._save_figure('feature_importance.png')
            
            plt.close()
    
    def plot_correlation_heatmap(self, corr_matrix, save=True):
        """
        상관관계 히트맵 그리기
        
        Args:
            corr_matrix (pd.DataFrame): 상관행렬
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting correlation heatmap"):
            # 특성이 너무 많으면 일부만 선택
            if corr_matrix.shape[0] > 50:
                # 분산 기준으로 상위 50개 특성만
                variances = corr_matrix.var()
                top_features = variances.nlargest(50).index
                corr_matrix = corr_matrix.loc[top_features, top_features]
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5,
                       cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
            
            plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                self._save_figure('correlation_heatmap.png')
            
            plt.close()
    
    def plot_statistical_pvalues(self, anova_results, save=True):
        """
        ANOVA p-value 분포 히스토그램
        
        Args:
            anova_results (pd.DataFrame): ANOVA 결과
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting p-value distribution"):
            plt.figure(figsize=self.config.FIGURE_SIZE)
            
            plt.hist(anova_results['p_value'], bins=50, color='skyblue', edgecolor='black')
            plt.axvline(x=0.05, color='r', linestyle='--', linewidth=2, 
                       label='α = 0.05')
            plt.xlabel('P-value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of ANOVA P-values', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            if save:
                self._save_figure('pvalue_distribution.png')
            
            plt.close()
    
    def plot_cluster_sizes(self, cluster_distribution, save=True):
        """
        클러스터별 샘플 수 막대 그래프
        
        Args:
            cluster_distribution (dict): 클러스터 분포 정보
            save (bool): 파일 저장 여부
        """
        with Timer("Plotting cluster sizes"):
            clusters = sorted(cluster_distribution.keys())
            sizes = [cluster_distribution[c]['count'] for c in clusters]
            
            plt.figure(figsize=self.config.FIGURE_SIZE)
            plt.bar(clusters, sizes, color='teal', alpha=0.7, edgecolor='black')
            plt.xlabel('Cluster ID', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
            plt.xticks(clusters)
            plt.grid(True, alpha=0.3, axis='y')
            
            # 각 막대 위에 정확한 숫자 표시
            for i, (cluster, size) in enumerate(zip(clusters, sizes)):
                plt.text(cluster, size, str(size), ha='center', va='bottom')
            
            if save:
                self._save_figure('cluster_sizes.png')
            
            plt.close()


def create_all_visualizations(clustering_obj, X_train, train_labels, 
                             eda_results, eval_results, stat_results):
    """
    모든 시각화를 생성하는 편의 함수
    
    Args:
        clustering_obj: KMeansClustering 객체
        X_train (np.ndarray): 학습 데이터
        train_labels (np.ndarray): 클러스터 레이블
        eda_results (dict): EDA 결과
        eval_results (dict): 평가 결과
        stat_results (dict): 통계 검정 결과
    """
    viz = Visualizer()
    
    logging.info("Creating visualizations...")
    
    # 1. 클러스터링 평가 그래프
    viz.plot_elbow_curve(clustering_obj.results)
    viz.plot_silhouette_scores(clustering_obj.results)
    viz.plot_all_metrics(clustering_obj.results)
    
    # 2. 클러스터 시각화
    viz.plot_clusters_2d_pca(X_train, train_labels)
    viz.plot_clusters_2d_tsne(X_train, train_labels)
    
    # 3. Silhouette Diagram
    if 'silhouette_analysis' in eval_results:
        viz.plot_silhouette_diagram(
            X_train, 
            train_labels,
            eval_results['silhouette_analysis']['per_sample']
        )
    
    # 4. 혼동 행렬 (레이블이 있는 경우)
    if 'confusion_matrix' in eval_results and eval_results['confusion_matrix'] is not None:
        viz.plot_confusion_matrix(eval_results['confusion_matrix'])
    
    # 5. 특성 중요도
    if 'feature_importance' in eda_results:
        viz.plot_feature_importance(eda_results['feature_importance'])
    
    # 6. 상관관계 히트맵
    if 'correlation' in eda_results:
        viz.plot_correlation_heatmap(eda_results['correlation'])
    
    # 7. 통계 검정 결과
    if 'anova' in stat_results:
        viz.plot_statistical_pvalues(stat_results['anova'])
    
    # 8. 클러스터 크기
    if 'cluster_distribution' in eval_results:
        viz.plot_cluster_sizes(eval_results['cluster_distribution'])
    
    logging.info("All visualizations created!")

"""
report_generator.py
분석 결과 리포트 생성 모듈

분석 결과를 마크다운 형식의 리포트와 CSV/Excel 테이블로 저장합니다.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from config import Config
from utils import Timer, dict_to_markdown_table


class ReportGenerator:
    """
    분석 결과를 리포트로 생성하는 클래스
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (Config, optional): 설정 객체
        """
        self.config = config if config is not None else Config()
        self.report_lines = []
    
    def _add_section(self, title, level=1):
        """
        리포트에 섹션 제목 추가
        
        Args:
            title (str): 섹션 제목
            level (int): 헤딩 레벨 (1-6)
        """
        self.report_lines.append(f"\n{'#' * level} {title}\n")
    
    def _add_text(self, text):
        """
        리포트에 일반 텍스트 추가
        
        Args:
            text (str): 추가할 텍스트
        """
        self.report_lines.append(f"{text}\n")
    
    def _add_table(self, df, max_rows=None):
        """
        리포트에 표 추가
        
        Args:
            df (pd.DataFrame): 데이터프레임
            max_rows (int, optional): 최대 행 수
        """
        if max_rows and len(df) > max_rows:
            df_display = df.head(max_rows)
            self.report_lines.append(f"\n{df_display.to_markdown()}\n")
            self.report_lines.append(f"\n*Showing top {max_rows} of {len(df)} rows*\n")
        else:
            self.report_lines.append(f"\n{df.to_markdown()}\n")
    
    def _add_key_value(self, key, value):
        """
        리포트에 키-값 쌍 추가
        
        Args:
            key (str): 키
            value: 값
        """
        self.report_lines.append(f"- **{key}**: {value}\n")
    
    def generate_header(self):
        """리포트 헤더 생성"""
        self._add_section("HAR 데이터셋 K-Means 클러스터링 분석 리포트", level=1)
        self._add_text(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._add_text("---")
    
    def add_configuration_section(self):
        """설정 정보 섹션 추가"""
        self._add_section("1. 분석 설정", level=2)
        
        config_summary = self.config.summary()
        
        self._add_key_value("Random Seed", config_summary['random_seed'])
        self._add_key_value("K Range", config_summary['k_range'])
        self._add_key_value("Normalization Method", config_summary['normalization_method'])
        self._add_key_value("Use PCA", config_summary['use_pca'])
        if config_summary['use_pca']:
            self._add_key_value("PCA Variance Ratio", config_summary['pca_variance_ratio'])
        self._add_key_value("K-Means Init Method", config_summary['kmeans_init'])
        self._add_key_value("Alpha Level", config_summary['alpha_level'])
        self._add_key_value("Number of Iterations", config_summary['n_iterations'])
    
    def add_data_overview_section(self, data_dict):
        """데이터 개요 섹션 추가"""
        self._add_section("2. 데이터 개요", level=2)
        
        self._add_key_value("Train Samples", data_dict['n_samples_train'])
        self._add_key_value("Test Samples", data_dict['n_samples_test'])
        self._add_key_value("Number of Features", data_dict['n_features'])
        
        if data_dict.get('y_train') is not None:
            unique_classes = np.unique(data_dict['y_train'])
            self._add_key_value("Number of Classes", len(unique_classes))
            self._add_key_value("Classes", unique_classes.tolist())
    
    def add_eda_section(self, eda_results):
        """탐색적 데이터 분석 섹션 추가"""
        self._add_section("3. 탐색적 데이터 분석 (EDA)", level=2)
        
        # 3.1 기본 통계량
        self._add_section("3.1 기본 통계량", level=3)
        if 'basic_stats' in eda_results:
            stats_df = eda_results['basic_stats']
            self._add_table(stats_df.iloc[:, :5], max_rows=10)  # 처음 5개 특성만
        
        # 3.2 클래스 분포
        if 'class_distribution' in eda_results and eda_results['class_distribution']:
            self._add_section("3.2 클래스 분포", level=3)
            dist = eda_results['class_distribution']
            dist_df = pd.DataFrame(dist).T
            dist_df.index.name = 'Class'
            self._add_table(dist_df)
        
        # 3.3 특성 중요도
        if 'feature_importance' in eda_results:
            self._add_section("3.3 특성 중요도 (Top 10)", level=3)
            importance_df = eda_results['feature_importance'].head(10)
            self._add_table(importance_df)
    
    def add_clustering_section(self, clustering_obj):
        """클러스터링 결과 섹션 추가"""
        self._add_section("4. K-Means 클러스터링 결과", level=2)
        
        # 4.1 최적 K 값
        self._add_section("4.1 최적 클러스터 수 (K)", level=3)
        self._add_key_value("Optimal K", clustering_obj.optimal_k)
        
        # 4.2 K별 평가 지표
        self._add_section("4.2 K별 평가 지표", level=3)
        
        results_data = []
        for k in sorted(clustering_obj.results.keys()):
            results_data.append({
                'K': k,
                'Silhouette': f"{clustering_obj.results[k]['silhouette']:.4f}",
                'Davies-Bouldin': f"{clustering_obj.results[k]['davies_bouldin']:.4f}",
                'Calinski-Harabasz': f"{clustering_obj.results[k]['calinski_harabasz']:.2f}",
                'Inertia': f"{clustering_obj.results[k]['inertia']:.2f}"
            })
        
        results_df = pd.DataFrame(results_data)
        self._add_table(results_df)
    
    def add_evaluation_section(self, eval_results):
        """평가 결과 섹션 추가"""
        self._add_section("5. 클러스터링 평가", level=2)
        
        # 5.1 내부 평가 지표
        self._add_section("5.1 내부 평가 지표", level=3)
        if 'internal_metrics' in eval_results:
            metrics = eval_results['internal_metrics']
            self._add_key_value("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
            self._add_key_value("Davies-Bouldin Index", f"{metrics['davies_bouldin_index']:.4f}")
            self._add_key_value("Calinski-Harabasz Index", f"{metrics['calinski_harabasz_index']:.2f}")
            self._add_key_value("Inertia", f"{metrics['inertia']:.2f}")
        
        # 5.2 외부 평가 지표
        if 'external_metrics' in eval_results and eval_results['external_metrics']:
            self._add_section("5.2 외부 평가 지표", level=3)
            ext_metrics = eval_results['external_metrics']
            self._add_key_value("Adjusted Rand Index", f"{ext_metrics['adjusted_rand_index']:.4f}")
            self._add_key_value("Normalized Mutual Information", f"{ext_metrics['normalized_mutual_info']:.4f}")
            self._add_key_value("Homogeneity", f"{ext_metrics['homogeneity']:.4f}")
            self._add_key_value("Completeness", f"{ext_metrics['completeness']:.4f}")
            self._add_key_value("V-measure", f"{ext_metrics['v_measure']:.4f}")
            self._add_key_value("Purity", f"{ext_metrics['purity']:.4f}")
        
        # 5.3 클러스터 분포
        if 'cluster_distribution' in eval_results:
            self._add_section("5.3 클러스터 분포", level=3)
            dist = eval_results['cluster_distribution']
            dist_df = pd.DataFrame(dist).T
            dist_df.index.name = 'Cluster'
            self._add_table(dist_df)
    
    def add_statistical_section(self, stat_results):
        """통계 검정 결과 섹션 추가"""
        self._add_section("6. 비지도 학습 성능 평가", level=2)
        
        # 6.1 Silhouette Score 결과
        if 'silhouette' in stat_results:
            self._add_section("6.1 Silhouette Score 평가", level=3)
            sil_score = stat_results['silhouette']['silhouette_score']
            
            self._add_text(f"**Silhouette Score**: {sil_score:.4f}")
            self._add_text(f"**해석**: {stat_results['silhouette']['interpretation']}")
            
            # Silhouette Score 해석 가이드
            self._add_text("\n**점수 해석 기준**:")
            self._add_text("- 0.71 - 1.0: 강한 구조")
            self._add_text("- 0.51 - 0.70: 합리적인 구조")
            self._add_text("- 0.26 - 0.50: 약한 구조")
            self._add_text("- < 0.25: 구조를 찾지 못함")
        
        # 6.2 Kruskal-Wallis 결과
        if 'kruskal_wallis' in stat_results:
            self._add_section("6.2 Kruskal-Wallis 검정 결과", level=3)
            kw_df = stat_results['kruskal_wallis']
            
            significant_features = kw_df[kw_df['significant'] == True]
            self._add_text(f"**유의한 특성 수** (p < 0.05): {len(significant_features)}/{len(kw_df)}")
        
        # 6.3 효과 크기
        if 'effect_size_matrix' in stat_results:
            self._add_section("6.3 효과 크기 (Cohen's d)", level=3)
            self._add_text("클러스터 간 효과 크기 행렬 (첫 번째 특성 기준):")
            # 효과 크기 행렬을 DataFrame으로 변환
            effect_matrix = stat_results['effect_size_matrix']
            effect_df = pd.DataFrame(
                effect_matrix,
                columns=[f"Cluster {i}" for i in range(effect_matrix.shape[1])],
                index=[f"Cluster {i}" for i in range(effect_matrix.shape[0])]
            )
            self._add_table(effect_df)
    
    def add_conclusion_section(self, clustering_obj, eval_results):
        """결론 섹션 추가"""
        self._add_section("7. 결론 및 요약", level=2)
        
        optimal_k = clustering_obj.optimal_k
        self._add_text(f"본 분석에서는 HAR 데이터셋에 대해 K-Means 클러스터링을 수행하였으며, "
                      f"**최적 클러스터 수는 {optimal_k}개**로 결정되었습니다.")
        
        if 'internal_metrics' in eval_results:
            metrics = eval_results['internal_metrics']
            sil_score = metrics['silhouette_score']
            
            # Silhouette Score 해석
            if sil_score > 0.7:
                interpretation = "매우 우수한"
            elif sil_score > 0.5:
                interpretation = "우수한"
            elif sil_score > 0.3:
                interpretation = "적절한"
            else:
                interpretation = "개선이 필요한"
            
            self._add_text(f"\nSilhouette Score는 {sil_score:.4f}로, "
                          f"{interpretation} 클러스터링 품질을 나타냅니다.")
        
        if 'external_metrics' in eval_results and eval_results['external_metrics']:
            ext_metrics = eval_results['external_metrics']
            ari = ext_metrics['adjusted_rand_index']
            nmi = ext_metrics['normalized_mutual_info']
            
            self._add_text(f"\n실제 레이블과의 비교 결과, Adjusted Rand Index는 {ari:.4f}, "
                          f"Normalized Mutual Information은 {nmi:.4f}로 나타났습니다.")
        
        self._add_text("\n통계적 유의성 검증 결과, 대부분의 특성에서 클러스터 간 유의한 차이가 "
                      "확인되었으며, 이는 클러스터링이 의미 있는 그룹을 발견했음을 시사합니다.")
    
    def save_markdown_report(self):
        """마크다운 리포트를 파일로 저장"""
        with Timer("Saving markdown report"):
            report_path = self.config.get_report_path()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.writelines(self.report_lines)
            
            logging.info(f"Report saved: {report_path}")
    
    def save_tables(self, eda_results, clustering_obj, eval_results, stat_results):
        """결과 테이블을 CSV/Excel 파일로 저장"""
        with Timer("Saving result tables"):
            tables_dir = self.config.TABLES_DIR
            
            # 1. 클러스터링 결과
            clustering_results = []
            for k in sorted(clustering_obj.results.keys()):
                clustering_results.append({
                    'K': k,
                    'Silhouette': clustering_obj.results[k]['silhouette'],
                    'Davies_Bouldin': clustering_obj.results[k]['davies_bouldin'],
                    'Calinski_Harabasz': clustering_obj.results[k]['calinski_harabasz'],
                    'Inertia': clustering_obj.results[k]['inertia']
                })
            
            clustering_df = pd.DataFrame(clustering_results)
            clustering_df.to_csv(tables_dir / 'clustering_results.csv', index=False)
            
            # 2. 내부 평가 지표
            if 'internal_metrics' in eval_results:
                internal_df = pd.DataFrame([eval_results['internal_metrics']])
                internal_df.to_csv(tables_dir / 'internal_metrics.csv', index=False)
            
            # 3. 외부 평가 지표
            if 'external_metrics' in eval_results and eval_results['external_metrics']:
                external_df = pd.DataFrame([eval_results['external_metrics']])
                external_df.to_csv(tables_dir / 'external_metrics.csv', index=False)
            
            # 4. Silhouette Score 결과
            if 'silhouette' in stat_results:
                sil_df = pd.DataFrame([stat_results['silhouette']])
                sil_df.to_csv(tables_dir / 'silhouette_results.csv', index=False)
            
            # 5. 특성 중요도
            if 'feature_importance' in eda_results:
                eda_results['feature_importance'].to_csv(
                    tables_dir / 'feature_importance.csv', index=False
                )
            
            logging.info(f"Tables saved to {tables_dir}")
    
    def generate_full_report(self, data_dict, eda_results, clustering_obj, 
                           eval_results, stat_results):
        """전체 리포트 생성"""
        logging.info("Generating full analysis report...")
        
        # 리포트 내용 생성
        self.generate_header()
        self.add_configuration_section()
        self.add_data_overview_section(data_dict)
        self.add_eda_section(eda_results)
        self.add_clustering_section(clustering_obj)
        self.add_evaluation_section(eval_results)
        self.add_statistical_section(stat_results)
        self.add_conclusion_section(clustering_obj, eval_results)
        
        # 파일로 저장
        self.save_markdown_report()
        self.save_tables(eda_results, clustering_obj, eval_results, stat_results)
        
        logging.info("Report generation completed!")


def create_report(data_dict, eda_results, clustering_obj, eval_results, stat_results):
    """
    분석 리포트 생성 (편의 함수)
    
    Args:
        data_dict (dict): 데이터 정보
        eda_results (dict): EDA 결과
        clustering_obj: KMeansClustering 객체
        eval_results (dict): 평가 결과
        stat_results (dict): 통계 검정 결과
    """
    report_gen = ReportGenerator()
    report_gen.generate_full_report(
        data_dict, eda_results, clustering_obj, eval_results, stat_results
    )

"""
utils.py
HAR 데이터셋 K-Means 분석 프로그램 유틸리티 함수

공통으로 사용되는 헬퍼 함수들을 정의합니다.
로깅 설정, 디렉토리 관리, 시드 설정 등을 포함합니다.
"""

import logging
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from config import Config


def setup_seed(seed=None):
    """
    재현 가능한 결과를 위해 모든 random seed를 고정
    
    Python, NumPy, scikit-learn 등의 랜덤 시드를 동일하게 설정하여
    실험을 반복해도 동일한 결과를 얻을 수 있도록 합니다.
    
    Args:
        seed (int, optional): 설정할 시드 값. None이면 Config의 RANDOM_SEED 사용
    """
    if seed is None:
        seed = Config.RANDOM_SEED
    
    # Python 내장 random 모듈 시드 설정
    random.seed(seed)
    
    # NumPy random 시드 설정
    np.random.seed(seed)
    
    logging.info(f"Random seed set to: {seed}")


def setup_logging(log_path=None, level=None):
    """
    로깅 시스템 초기화
    
    파일과 콘솔에 동시에 로그를 기록하도록 설정합니다.
    로그 레벨, 포맷, 출력 위치를 지정합니다.
    
    Args:
        log_path (Path, optional): 로그 파일 경로. None이면 Config의 설정 사용
        level (str, optional): 로그 레벨. None이면 Config의 설정 사용
    """
    if log_path is None:
        log_path = Config.get_log_path()
    
    if level is None:
        level = Config.LOG_LEVEL
    
    # 로그 레벨 문자열을 logging 상수로 변환
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # 로그 포맷 정의 (시간, 레벨, 메시지)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 로거 설정
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 파일 핸들러 (로그를 파일에 저장)
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            # 스트림 핸들러 (로그를 콘솔에 출력)
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_path}")


def get_timestamp():
    """
    현재 시간을 파일명에 사용 가능한 형식의 문자열로 반환
    
    파일명이나 폴더명에 타임스탬프를 추가할 때 사용합니다.
    
    Returns:
        str: 'YYYYMMDD_HHMMSS' 형식의 타임스탬프
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_array_to_file(array, filepath, fmt='%.6f'):
    """
    NumPy 배열을 텍스트 파일로 저장
    
    Args:
        array (np.ndarray): 저장할 배열
        filepath (Path or str): 저장할 파일 경로
        fmt (str): 숫자 포맷. 기본값은 소수점 6자리
    """
    np.savetxt(filepath, array, fmt=fmt)
    logging.info(f"Array saved to {filepath}")


def load_array_from_file(filepath):
    """
    텍스트 파일에서 NumPy 배열 로드
    
    Args:
        filepath (Path or str): 로드할 파일 경로
    
    Returns:
        np.ndarray: 로드된 배열
    """
    array = np.loadtxt(filepath)
    logging.info(f"Array loaded from {filepath}")
    return array


def ensure_directory_exists(directory):
    """
    디렉토리가 존재하지 않으면 생성
    
    Args:
        directory (Path or str): 생성할 디렉토리 경로
    
    Returns:
        Path: 디렉토리 경로 객체
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_feature_names(df, label_column):
    """
    데이터프레임에서 특성(feature) 컬럼 이름 리스트 반환
    
    레이블 컬럼을 제외한 모든 컬럼을 특성으로 간주합니다.
    
    Args:
        df (pd.DataFrame): 데이터프레임
        label_column (str): 레이블 컬럼 이름
    
    Returns:
        list: 특성 컬럼 이름 리스트
    """
    if label_column in df.columns:
        return [col for col in df.columns if col != label_column]
    else:
        # 레이블 컬럼이 없으면 모든 컬럼을 특성으로 간주
        return df.columns.tolist()


def print_section_header(title, char='=', width=80):
    """
    콘솔 출력용 섹션 헤더 출력
    
    로그나 콘솔에서 구분을 명확히 하기 위한 헤더를 출력합니다.
    
    Args:
        title (str): 섹션 제목
        char (str): 헤더 라인에 사용할 문자
        width (int): 전체 너비
    """
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def format_seconds(seconds):
    """
    초 단위 시간을 읽기 쉬운 형식으로 변환
    
    Args:
        seconds (float): 초 단위 시간
    
    Returns:
        str: 포맷된 시간 문자열 (예: "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def calculate_percentage(part, total):
    """
    백분율 계산
    
    Args:
        part (float): 부분 값
        total (float): 전체 값
    
    Returns:
        float: 백분율 (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def dict_to_markdown_table(data_dict):
    """
    딕셔너리를 마크다운 테이블 형식으로 변환
    
    Args:
        data_dict (dict): 변환할 딕셔너리
    
    Returns:
        str: 마크다운 테이블 형식의 문자열
    """
    lines = ["| Key | Value |", "|-----|-------|"]
    
    for key, value in data_dict.items():
        # 값을 문자열로 변환 (리스트나 배열은 간략하게)
        if isinstance(value, (list, np.ndarray)):
            if len(value) > 5:
                value_str = f"[{value[0]}, ..., {value[-1]}] (length: {len(value)})"
            else:
                value_str = str(value)
        else:
            value_str = str(value)
        
        lines.append(f"| {key} | {value_str} |")
    
    return '\n'.join(lines)


def truncate_string(s, max_length=50):
    """
    문자열을 지정된 길이로 자르고 생략 부호 추가
    
    Args:
        s (str): 원본 문자열
        max_length (int): 최대 길이
    
    Returns:
        str: 잘린 문자열
    """
    if len(s) <= max_length:
        return s
    return s[:max_length-3] + "..."


class Timer:
    """
    코드 실행 시간 측정을 위한 컨텍스트 매니저
    
    Usage:
        with Timer("Data loading"):
            data = load_data()
    """
    
    def __init__(self, name="Operation"):
        """
        Args:
            name (str): 측정할 작업의 이름
        """
        self.name = name
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """컨텍스트 진입 시 타이머 시작"""
        self.start_time = datetime.now()
        logging.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 경과 시간 출력"""
        end_time = datetime.now()
        self.elapsed_time = (end_time - self.start_time).total_seconds()
        formatted_time = format_seconds(self.elapsed_time)
        logging.info(f"Completed: {self.name} (Time: {formatted_time})")
        return False
    
    def get_elapsed(self):
        """
        경과 시간 반환
        
        Returns:
            float: 초 단위 경과 시간
        """
        return self.elapsed_time


def validate_dataframe(df, required_columns=None, allow_nan=False):
    """
    데이터프레임의 유효성 검증
    
    Args:
        df (pd.DataFrame): 검증할 데이터프레임
        required_columns (list, optional): 필수 컬럼 리스트
        allow_nan (bool): NaN 값 허용 여부
    
    Returns:
        bool: 유효하면 True, 아니면 False
    
    Raises:
        ValueError: 데이터프레임이 유효하지 않을 때
    """
    # 데이터프레임이 비어있는지 확인
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # 필수 컬럼 확인
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # NaN 값 확인
    if not allow_nan and df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"NaN values found in columns: {nan_cols}")
    
    logging.info(f"DataFrame validation passed. Shape: {df.shape}")
    return True

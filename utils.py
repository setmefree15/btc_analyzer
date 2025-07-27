"""
공통 유틸리티 함수
"""
import os
import json
import pickle
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# 로깅 설정
def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 파일 관리
def save_json(data: Dict, filepath: str) -> None:
    """JSON 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Dict:
    """JSON 파일 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_pickle(data: Any, filepath: str) -> None:
    """Pickle 파일 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath: str) -> Any:
    """Pickle 파일 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# 데이터 처리
def resample_data(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """데이터 리샘플링"""
    rule_map = {
        '15m': '15T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    if source_tf == target_tf:
        return df.copy()
    
    rule = rule_map.get(target_tf)
    if not rule:
        raise ValueError(f"지원하지 않는 타임프레임: {target_tf}")
    
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled

def calculate_returns(prices: pd.Series) -> pd.Series:
    """수익률 계산"""
    return prices.pct_change().fillna(0)

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """누적 수익률 계산"""
    return (1 + returns).cumprod() - 1

# 성과 측정
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """샤프 비율 계산"""
    excess_returns = returns - risk_free_rate / 252  # 일간 무위험 수익률
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """최대 낙폭 계산"""
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / (1 + running_max)
    return drawdown.min()

def calculate_win_rate(trades: List[Dict]) -> float:
    """승률 계산"""
    if not trades:
        return 0
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    return winning_trades / len(trades)

# 시간 관련
def get_datetime_range(start_date: str, end_date: str) -> List[datetime]:
    """날짜 범위 생성"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return pd.date_range(start=start, end=end, freq='D').tolist()

def format_timeframe(tf: str) -> str:
    """타임프레임 포맷팅"""
    tf_map = {
        '15m': '15분',
        '1h': '1시간',
        '4h': '4시간',
        '1d': '일봉'
    }
    return tf_map.get(tf, tf)

# 데이터 검증
def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """OHLCV 데이터 검증"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 필수 컬럼 확인
    if not all(col in df.columns for col in required_columns):
        return False
    
    # 데이터 무결성 확인
    if df.isnull().any().any():
        return False
    
    # 가격 관계 확인 (high >= low, high >= open/close, low <= open/close)
    if not (df['high'] >= df['low']).all():
        return False
    if not (df['high'] >= df[['open', 'close']].max(axis=1)).all():
        return False
    if not (df['low'] <= df[['open', 'close']].min(axis=1)).all():
        return False
    
    return True

# 진행 상태 표시
def format_progress(current: int, total: int, prefix: str = '') -> str:
    """진행 상태 포맷팅"""
    percentage = current / total * 100 if total > 0 else 0
    return f"{prefix} {current}/{total} ({percentage:.1f}%)"

# 숫자 포맷팅
def format_number(num: float, decimals: int = 2) -> str:
    """숫자 포맷팅"""
    return f"{num:,.{decimals}f}"

def format_percentage(num: float, decimals: int = 2) -> str:
    """퍼센트 포맷팅"""
    return f"{num * 100:.{decimals}f}%"

# 에러 처리
def safe_divide(a: float, b: float, default: float = 0) -> float:
    """안전한 나눗셈"""
    return a / b if b != 0 else default
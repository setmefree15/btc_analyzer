"""
BTC 분석 시스템 설정 파일
"""
import os
from datetime import datetime

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 거래소 설정
EXCHANGE = 'binance'
SYMBOL = 'BTC/USDT'

# 타임프레임 설정
TIMEFRAMES = {
    '15m': '15분',
    '1h': '1시간',
    '4h': '4시간',
    '1d': '일봉'
}

# 데이터 수집 설정
START_DATE = '2021-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# 기술적 지표 파라미터
INDICATOR_PARAMS = {
    'RSI': {
        'periods': [9, 14, 21]
    },
    'EMA': {
        'periods': [10, 20, 50]
    },
    'SMA': {
        'periods': [10, 20, 50]
    },
    'BB': {
        'period': 20,
        'std_devs': [1.5, 2.0, 2.5]
    },
    'MACD': {
        'configs': [(12, 26, 9), (8, 21, 5)]
    },
    'STOCH': {
        'configs': [(14, 3, 3), (5, 3, 3)]
    }
}

# 백테스팅 설정
BACKTEST_CONFIG = {
    'initial_capital': 10000,
    'commission': 0.001,  # 0.1%
    'slippage': 0.001,    # 0.1%
    'risk_per_trade': 0.02  # 2%
}

# 프랙탈 분석 설정
FRACTAL_CONFIG = {
    'window_size': 50,  # 50봉 단위
    'min_similarity': 0.7,  # 최소 유사도 70%
    'top_matches': 10  # 상위 10개 매칭
}

# 스트림릿 설정
STREAMLIT_CONFIG = {
    'page_title': '🚀 BTC 전용 분석 시스템',
    'page_icon': '🚀',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# 캐시 설정
CACHE_TTL = 3600  # 1시간

# 로깅 설정
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
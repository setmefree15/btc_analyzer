"""
기술적 지표 계산 모듈 (TA-Lib 버전)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
import warnings
warnings.filterwarnings('ignore')

import config
import utils

logger = utils.setup_logger(__name__)

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self):
        """초기화"""
        self.indicators_config = config.INDICATOR_PARAMS
        
    def calculate_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """RSI 계산"""
        if periods is None:
            periods = self.indicators_config['RSI']['periods']
        
        for period in periods:
            col_name = f'RSI_{period}'
            df[col_name] = talib.RSI(df['close'], timeperiod=period)
            logger.debug(f"RSI {period} 계산 완료")
        
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame, 
                                  ema_periods: List[int] = None,
                                  sma_periods: List[int] = None) -> pd.DataFrame:
        """이동평균 계산 (EMA, SMA)"""
        if ema_periods is None:
            ema_periods = self.indicators_config['EMA']['periods']
        if sma_periods is None:
            sma_periods = self.indicators_config['SMA']['periods']
        
        # EMA 계산
        for period in ema_periods:
            col_name = f'EMA_{period}'
            df[col_name] = talib.EMA(df['close'], timeperiod=period)
            logger.debug(f"EMA {period} 계산 완료")
        
        # SMA 계산
        for period in sma_periods:
            col_name = f'SMA_{period}'
            df[col_name] = talib.SMA(df['close'], timeperiod=period)
            logger.debug(f"SMA {period} 계산 완료")
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, 
                                  period: int = None,
                                  std_devs: List[float] = None) -> pd.DataFrame:
        """볼린저 밴드 계산"""
        if period is None:
            period = self.indicators_config['BB']['period']
        if std_devs is None:
            std_devs = self.indicators_config['BB']['std_devs']
        
        for std_dev in std_devs:
            upper, middle, lower = talib.BBANDS(
                df['close'], 
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
            
            # 컬럼명 변경
            std_str = str(std_dev).replace('.', '_')
            df[f'BB_upper_{std_str}'] = upper
            df[f'BB_middle_{std_str}'] = middle
            df[f'BB_lower_{std_str}'] = lower
            
            # 밴드폭과 %B 계산
            df[f'BB_bandwidth_{std_str}'] = (upper - lower) / middle
            df[f'BB_percent_{std_str}'] = (df['close'] - lower) / (upper - lower)
            
            logger.debug(f"볼린저 밴드 (std={std_dev}) 계산 완료")
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame, 
                       configs: List[Tuple[int, int, int]] = None) -> pd.DataFrame:
        """MACD 계산"""
        if configs is None:
            configs = self.indicators_config['MACD']['configs']
        
        for fast, slow, signal in configs:
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'],
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            
            # 컬럼명 설정
            suffix = f'{fast}_{slow}_{signal}'
            df[f'MACD_{suffix}'] = macd
            df[f'MACD_signal_{suffix}'] = macd_signal
            df[f'MACD_hist_{suffix}'] = macd_hist
            
            logger.debug(f"MACD ({fast},{slow},{signal}) 계산 완료")
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame,
                            configs: List[Tuple[int, int, int]] = None) -> pd.DataFrame:
        """스토캐스틱 계산"""
        if configs is None:
            configs = self.indicators_config['STOCH']['configs']
        
        for fastk_period, slowk_period, slowd_period in configs:
            slowk, slowd = talib.STOCH(
                df['high'], 
                df['low'], 
                df['close'],
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowk_matype=0,
                slowd_period=slowd_period,
                slowd_matype=0
            )
            
            # 컬럼명 설정
            suffix = f'{fastk_period}_{slowk_period}_{slowd_period}'
            df[f'STOCH_K_{suffix}'] = slowk
            df[f'STOCH_D_{suffix}'] = slowd
            
            logger.debug(f"스토캐스틱 ({fastk_period},{slowk_period},{slowd_period}) 계산 완료")
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR (Average True Range) 계산"""
        df[f'ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        logger.debug(f"ATR {period} 계산 완료")
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 지표 계산"""
        # OBV (On Balance Volume)
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        
        # Volume SMA
        df['Volume_SMA_20'] = talib.SMA(df['volume'], timeperiod=20)
        
        # AD (Accumulation/Distribution)
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # MFI (Money Flow Index)
        df['MFI_14'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Volume Rate of Change
        df['Volume_ROC'] = talib.ROC(df['volume'], timeperiod=10)
        
        logger.debug("거래량 지표 계산 완료")
        return df
    
    def calculate_additional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """추가 지표 계산"""
        # ADX (Average Directional Index)
        df['ADX_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['DI+_14'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['DI-_14'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['CCI_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Williams %R
        df['WILLR_14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ROC (Rate of Change)
        df['ROC_10'] = talib.ROC(df['close'], timeperiod=10)
        
        # Momentum
        df['MOM_10'] = talib.MOM(df['close'], timeperiod=10)
        
        # TRIX
        df['TRIX_15'] = talib.TRIX(df['close'], timeperiod=15)
        
        # Ultimate Oscillator
        df['ULTOSC'] = talib.ULTOSC(df['high'], df['low'], df['close'], 
                                     timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        # Aroon
        aroondown, aroonup = talib.AROON(df['high'], df['low'], timeperiod=14)
        df['AROON_DOWN_14'] = aroondown
        df['AROON_UP_14'] = aroonup
        df['AROONOSC_14'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)
        
        logger.debug("추가 지표 계산 완료")
        return df
    
    def calculate_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """캔들 패턴 인식 (TA-Lib의 고유 기능)"""
        # 주요 캔들 패턴
        df['CDL_DOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDL_HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDL_SHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDL_ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['CDL_MORNINGSTAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDL_EVENINGSTAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDL_HARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        df['CDL_THREEWHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        df['CDL_THREEBLACKCROWS'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        
        logger.debug("캔들 패턴 인식 완료")
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame, timeframe: str = '') -> pd.DataFrame:
        """모든 지표 계산"""
        logger.info(f"모든 지표 계산 시작 (타임프레임: {timeframe})")
        
        # 원본 데이터 복사
        df_indicators = df.copy()
        
        # 각 지표 계산
        df_indicators = self.calculate_rsi(df_indicators)
        df_indicators = self.calculate_moving_averages(df_indicators)
        df_indicators = self.calculate_bollinger_bands(df_indicators)
        df_indicators = self.calculate_macd(df_indicators)
        df_indicators = self.calculate_stochastic(df_indicators)
        df_indicators = self.calculate_atr(df_indicators)
        df_indicators = self.calculate_volume_indicators(df_indicators)
        df_indicators = self.calculate_additional_indicators(df_indicators)
        df_indicators = self.calculate_pattern_recognition(df_indicators)
        
        logger.info(f"모든 지표 계산 완료 - 총 {len(df_indicators.columns)} 컬럼")
        
        return df_indicators
    
    def add_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래 신호 추가"""
        # RSI 신호 (완화된 조건)
        df['RSI_14_oversold'] = (df['RSI_14'] < 35).astype(int)  # 30 -> 35
        df['RSI_14_overbought'] = (df['RSI_14'] > 65).astype(int)  # 70 -> 65
        
        # MACD 신호
        macd_col = 'MACD_12_26_9'
        signal_col = 'MACD_signal_12_26_9'
        if macd_col in df.columns and signal_col in df.columns:
            df['MACD_bullish'] = (
                (df[macd_col] > df[signal_col]) & 
                (df[macd_col].shift(1) <= df[signal_col].shift(1))
            ).astype(int)
            df['MACD_bearish'] = (
                (df[macd_col] < df[signal_col]) & 
                (df[macd_col].shift(1) >= df[signal_col].shift(1))
            ).astype(int)
        
        # 볼린저 밴드 신호
        bb_upper = 'BB_upper_2_0'
        bb_lower = 'BB_lower_2_0'
        if bb_upper in df.columns and bb_lower in df.columns:
            df['BB_squeeze'] = (
                (df[bb_upper] - df[bb_lower]) < 
                (df[bb_upper] - df[bb_lower]).rolling(20).mean() * 0.8
            ).astype(int)
            df['BB_breakout_up'] = (df['close'] > df[bb_upper]).astype(int)
            df['BB_breakout_down'] = (df['close'] < df[bb_lower]).astype(int)
        
        # 이동평균 크로스
        if 'EMA_10' in df.columns and 'EMA_20' in df.columns:
            df['EMA_golden_cross'] = (
                (df['EMA_10'] > df['EMA_20']) & 
                (df['EMA_10'].shift(1) <= df['EMA_20'].shift(1))
            ).astype(int)
            df['EMA_death_cross'] = (
                (df['EMA_10'] < df['EMA_20']) & 
                (df['EMA_10'].shift(1) >= df['EMA_20'].shift(1))
            ).astype(int)
        
        # ADX 트렌드 강도
        if 'ADX_14' in df.columns:
            df['ADX_strong_trend'] = (df['ADX_14'] > 25).astype(int)
            df['ADX_weak_trend'] = (df['ADX_14'] < 20).astype(int)
        
        # 스토캐스틱 신호 (완화된 조건)
        stoch_k = 'STOCH_K_14_3_3'
        stoch_d = 'STOCH_D_14_3_3'
        if stoch_k in df.columns and stoch_d in df.columns:
            df['STOCH_oversold'] = (df[stoch_k] < 30).astype(int)  # 20 -> 30
            df['STOCH_overbought'] = (df[stoch_k] > 70).astype(int)  # 80 -> 70
            df['STOCH_bullish'] = (
                (df[stoch_k] > df[stoch_d]) & 
                (df[stoch_k].shift(1) <= df[stoch_d].shift(1))
            ).astype(int)
        
        # MFI 신호 (완화된 조건)
        if 'MFI_14' in df.columns:
            df['MFI_oversold'] = (df['MFI_14'] < 30).astype(int)  # 20 -> 30
            df['MFI_overbought'] = (df['MFI_14'] > 70).astype(int)  # 80 -> 70
        
        logger.debug("거래 신호 추가 완료")
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """지표 요약 정보"""
        summary = {
            'total_indicators': len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]),
            'latest_values': {},
            'signals': {},
            'patterns': {}
        }
        
        # 최신 지표 값
        latest_idx = df.index[-1]
        
        # 주요 지표 최신 값
        key_indicators = ['RSI_14', 'MACD_12_26_9', 'ADX_14', 'ATR_14', 'MFI_14', 'CCI_20']
        for indicator in key_indicators:
            if indicator in df.columns:
                value = df.loc[latest_idx, indicator]
                if pd.notna(value):
                    summary['latest_values'][indicator] = value
        
        # 활성 신호
        signal_columns = [col for col in df.columns if any(signal in col for signal in ['oversold', 'overbought', 'bullish', 'bearish', 'cross', 'strong', 'weak'])]
        for signal in signal_columns:
            if signal in df.columns and df.loc[latest_idx, signal] == 1:
                summary['signals'][signal] = True
        
        # 캔들 패턴
        pattern_columns = [col for col in df.columns if col.startswith('CDL_')]
        for pattern in pattern_columns:
            if pattern in df.columns:
                value = df.loc[latest_idx, pattern]
                if pd.notna(value) and value != 0:
                    summary['patterns'][pattern] = int(value)
        
        return summary

# 지표 성능 테스트 함수
def test_indicators():
    """지표 계산 테스트"""
    from data_collector import BTCDataCollector
    
    print("=" * 60)
    print("기술적 지표 계산 테스트 (TA-Lib)")
    print("=" * 60)
    
    # TA-Lib 버전 확인
    print(f"\nTA-Lib 버전: {talib.__version__}")
    
    # 데이터 로드
    collector = BTCDataCollector()
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다. 먼저 데이터를 수집하세요.")
        return
    
    print(f"✅ 데이터 로드 완료: {len(df)} 개의 캔들")
    
    # 지표 계산
    indicators = TechnicalIndicators()
    
    print("\n지표 계산 중...")
    df_with_indicators = indicators.calculate_all_indicators(df[:1000], '1h')  # 처음 1000개만 테스트
    
    print(f"\n✅ 지표 계산 완료")
    print(f"원본 컬럼 수: {len(df.columns)}")
    print(f"지표 포함 컬럼 수: {len(df_with_indicators.columns)}")
    print(f"추가된 지표 수: {len(df_with_indicators.columns) - len(df.columns)}")
    
    # 거래 신호 추가
    df_with_signals = indicators.add_trading_signals(df_with_indicators)
    
    # 요약 정보
    summary = indicators.get_indicator_summary(df_with_signals)
    
    print("\n📊 지표 요약:")
    print(f"총 지표 수: {summary['total_indicators']}")
    
    print("\n최신 지표 값:")
    for indicator, value in summary['latest_values'].items():
        print(f"  {indicator}: {value:.2f}")
    
    if summary['signals']:
        print("\n🚨 활성 신호:")
        for signal in summary['signals']:
            print(f"  - {signal}")
    else:
        print("\n현재 활성 신호 없음")
    
    if summary['patterns']:
        print("\n🕯️ 캔들 패턴:")
        for pattern, value in summary['patterns'].items():
            direction = "강세" if value > 0 else "약세"
            print(f"  - {pattern}: {direction} ({value})")
    
    # 샘플 데이터 출력
    print("\n최근 5개 캔들의 주요 지표:")
    display_columns = ['close', 'RSI_14', 'MACD_12_26_9', 'ADX_14', 'MFI_14']
    available_columns = [col for col in display_columns if col in df_with_signals.columns]
    print(df_with_signals[available_columns].tail())

if __name__ == "__main__":
    test_indicators()
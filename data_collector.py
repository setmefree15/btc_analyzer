"""
BTC 데이터 수집 모듈
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Optional, Dict, List
from tqdm import tqdm

import config
import utils

logger = utils.setup_logger(__name__)

class BTCDataCollector:
    """BTC 데이터 수집 클래스"""
    
    def __init__(self):
        """초기화"""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        self.symbol = config.SYMBOL
        
    def fetch_ohlcv(self, timeframe: str, since: Optional[int] = None, limit: int = 1000) -> List:
        """OHLCV 데이터 가져오기"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            return ohlcv
        except Exception as e:
            logger.error(f"OHLCV 데이터 가져오기 실패: {e}")
            return []
    
    def fetch_btc_data(self, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """지정된 기간의 BTC 데이터 수집"""
        logger.info(f"데이터 수집 시작: {timeframe} ({start_date} ~ {end_date})")
        
        # 날짜를 timestamp로 변환
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        # 전체 데이터 저장할 리스트
        all_data = []
        
        # 타임프레임별 밀리초 계산
        tf_ms_map = {
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        tf_ms = tf_ms_map.get(timeframe, 60 * 60 * 1000)
        
        # 예상 데이터 개수 계산 (진행바용)
        total_candles = int((end_ts - start_ts) / tf_ms)
        
        # 진행바 설정
        pbar = tqdm(total=total_candles, desc=f"수집 중 ({timeframe})")
        
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                # 데이터 가져오기
                ohlcv = self.fetch_ohlcv(timeframe, since=current_ts, limit=1000)
                
                if not ohlcv:
                    logger.warning(f"데이터 없음: {pd.Timestamp(current_ts, unit='ms')}")
                    break
                
                # 리스트에 추가
                all_data.extend(ohlcv)
                
                # 마지막 타임스탬프 업데이트
                last_ts = ohlcv[-1][0]
                
                # 진행바 업데이트
                collected = len([d for d in all_data if d[0] <= end_ts])
                pbar.n = min(collected, total_candles)
                pbar.refresh()
                
                # 더 이상 새로운 데이터가 없으면 종료
                if last_ts == current_ts:
                    break
                    
                current_ts = last_ts + tf_ms
                
                # API 제한 고려
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"데이터 수집 중 오류: {e}")
                time.sleep(1)
                continue
        
        pbar.close()
        
        # DataFrame으로 변환
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 지정된 기간 내의 데이터만 필터링
            df = df[df.index <= end_date]
            
            # 중복 제거
            df = df[~df.index.duplicated(keep='first')]
            
            # 정렬
            df.sort_index(inplace=True)
            
            logger.info(f"수집 완료: {len(df)} 개의 캔들")
            return df
        else:
            logger.warning("수집된 데이터가 없습니다.")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, timeframe: str) -> str:
        """데이터를 CSV 파일로 저장"""
        if df.empty:
            logger.warning("저장할 데이터가 없습니다.")
            return ""
        
        filename = f"btc_usdt_{timeframe}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        
        try:
            df.to_csv(filepath)
            logger.info(f"데이터 저장 완료: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
            return ""
    
    def load_data(self, timeframe: str) -> pd.DataFrame:
        """저장된 데이터 로드"""
        filename = f"btc_usdt_{timeframe}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"파일이 존재하지 않습니다: {filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            logger.info(f"데이터 로드 완료: {len(df)} 개의 캔들")
            return df
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def update_data(self, timeframe: str) -> pd.DataFrame:
        """기존 데이터 업데이트 (최신 데이터 추가)"""
        # 기존 데이터 로드
        df_existing = self.load_data(timeframe)
        
        if df_existing.empty:
            # 기존 데이터가 없으면 전체 수집
            return self.fetch_btc_data(timeframe, config.START_DATE, config.END_DATE)
        
        # 마지막 데이터 시간 확인
        last_date = df_existing.index[-1]
        current_date = pd.Timestamp.now()
        
        # 하루 이상 차이나면 업데이트
        if (current_date - last_date).days >= 1:
            logger.info(f"데이터 업데이트 필요: {last_date} ~ {current_date}")
            
            # 새로운 데이터 수집
            df_new = self.fetch_btc_data(
                timeframe,
                last_date.strftime('%Y-%m-%d'),
                current_date.strftime('%Y-%m-%d')
            )
            
            if not df_new.empty:
                # 기존 데이터와 병합
                df_combined = pd.concat([df_existing, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                
                # 저장
                self.save_to_csv(df_combined, timeframe)
                return df_combined
        
        return df_existing
    
    def collect_all_timeframes(self, timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """모든 타임프레임 데이터 수집"""
        if timeframes is None:
            timeframes = list(config.TIMEFRAMES.keys())
        
        results = {}
        
        for tf in timeframes:
            logger.info(f"\n{'='*50}")
            logger.info(f"타임프레임: {config.TIMEFRAMES[tf]} ({tf})")
            
            # 데이터 수집
            df = self.fetch_btc_data(tf, config.START_DATE, config.END_DATE)
            
            if not df.empty:
                # 저장
                self.save_to_csv(df, tf)
                results[tf] = df
            else:
                logger.error(f"{tf} 데이터 수집 실패")
        
        return results

# 테스트 함수
def test_collector():
    """데이터 수집기 테스트"""
    collector = BTCDataCollector()
    
    # 테스트: 최근 100개 캔들 가져오기
    print("\n테스트: 최근 100개 1시간봉 데이터 가져오기")
    test_data = collector.fetch_ohlcv('1h', limit=100)
    
    if test_data:
        df = pd.DataFrame(test_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"✅ 성공: {len(df)} 개의 캔들 수집")
        print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"현재 가격: ${df['close'].iloc[-1]:,.2f}")
    else:
        print("❌ 실패: 데이터를 가져올 수 없습니다.")

if __name__ == "__main__":
    test_collector()
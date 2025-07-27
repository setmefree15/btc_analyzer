"""
백테스팅 디버깅 스크립트
"""
from data_collector import BTCDataCollector
from indicators import TechnicalIndicators
from backtester import StrategyBacktester
import pandas as pd

def debug_backtest():
    """백테스팅 디버깅"""
    print("=" * 60)
    print("백테스팅 디버깅")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드")
    collector = BTCDataCollector()
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return
    
    # 최근 1000개만 사용
    df = df.tail(1000).copy()
    print(f"✅ 데이터 로드: {len(df)} 개")
    
    # 2. 지표 계산
    print("\n2. 지표 계산")
    indicators = TechnicalIndicators()
    df = indicators.calculate_all_indicators(df, '1h')
    print(f"✅ 컬럼 수: {len(df.columns)}")
    
    # 3. RSI 통계 확인
    print("\n3. RSI 통계")
    rsi_clean = df['RSI_14'].dropna()
    print(f"RSI 범위: {rsi_clean.min():.1f} ~ {rsi_clean.max():.1f}")
    print(f"RSI 평균: {rsi_clean.mean():.1f}")
    print(f"RSI < 45: {(rsi_clean < 45).sum()}개 ({(rsi_clean < 45).sum() / len(rsi_clean) * 100:.1f}%)")
    print(f"RSI > 55: {(rsi_clean > 55).sum()}개 ({(rsi_clean > 55).sum() / len(rsi_clean) * 100:.1f}%)")
    
    # 4. 다양한 전략 테스트
    print("\n4. 전략 테스트")
    
    strategies = [
        {
            'name': 'RSI_45_55',
            'entry_rules': ['RSI_14 < 45'],
            'exit_rules': ['RSI_14 > 55']
        },
        {
            'name': 'RSI_50_CROSS',
            'entry_rules': ['RSI_14 < 50'],
            'exit_rules': ['RSI_14 > 50']
        },
        {
            'name': 'MACD_CROSS',
            'entry_rules': ['MACD_12_26_9 > MACD_signal_12_26_9'],
            'exit_rules': ['MACD_12_26_9 < MACD_signal_12_26_9']
        },
        {
            'name': 'BB_TOUCH',
            'entry_rules': ['close < BB_lower_2_0'],
            'exit_rules': ['close > BB_middle_2_0']
        }
    ]
    
    for strategy in strategies:
        print(f"\n테스트: {strategy['name']}")
        
        backtester = StrategyBacktester()
        results = backtester.backtest_strategy(df, strategy)
        
        print(f"거래 수: {results['total_trades']}")
        if 'entry_signals' in results:
            print(f"진입 신호: {results['entry_signals']}회")
            print(f"청산 신호: {results['exit_signals']}회")
        
        if results['total_trades'] > 0:
            print(f"수익률: {results['total_return_pct']:.1f}%")
            print(f"승률: {results['win_rate']*100:.1f}%")
    
    # 5. 신호 위치 확인
    print("\n5. RSI 신호 위치 확인 (처음 20개)")
    for i in range(100, min(120, len(df))):
        rsi_val = df['RSI_14'].iloc[i]
        if pd.notna(rsi_val):
            if rsi_val < 45:
                print(f"진입 신호: {df.index[i]} - RSI: {rsi_val:.1f}")
            elif rsi_val > 55:
                print(f"청산 신호: {df.index[i]} - RSI: {rsi_val:.1f}")

if __name__ == "__main__":
    debug_backtest()
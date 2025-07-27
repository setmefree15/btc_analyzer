"""
지표 계산 테스트 스크립트
"""
from data_collector import BTCDataCollector
from indicators import TechnicalIndicators
from indicator_combinations import IndicatorCombinations
import pandas as pd

def test_indicator_calculation():
    """지표 계산 테스트"""
    print("=" * 60)
    print("기술적 지표 계산 테스트")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드")
    collector = BTCDataCollector()
    
    # 1시간봉 데이터 로드
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다. 먼저 데이터를 수집하세요.")
        return
    
    print(f"✅ 데이터 로드 완료: {len(df):,} 개의 캔들")
    print(f"기간: {df.index.min()} ~ {df.index.max()}")
    
    # 최근 500개만 사용 (테스트용)
    df_test = df.tail(500).copy()
    
    # 2. 지표 계산
    print("\n2. 기술적 지표 계산")
    indicators = TechnicalIndicators()
    
    # 개별 지표 계산
    print("- RSI 계산...")
    df_test = indicators.calculate_rsi(df_test)
    
    print("- 이동평균 계산...")
    df_test = indicators.calculate_moving_averages(df_test)
    
    print("- 볼린저밴드 계산...")
    df_test = indicators.calculate_bollinger_bands(df_test)
    
    print("- MACD 계산...")
    df_test = indicators.calculate_macd(df_test)
    
    print("- 거래 신호 추가...")
    df_test = indicators.add_trading_signals(df_test)
    
    # 3. 결과 확인
    print(f"\n3. 계산 결과")
    print(f"원본 컬럼 수: 5 (open, high, low, close, volume)")
    print(f"지표 포함 컬럼 수: {len(df_test.columns)}")
    print(f"추가된 지표/신호: {len(df_test.columns) - 5}")
    
    # 최신 지표 값
    print("\n4. 최신 지표 값")
    latest = df_test.iloc[-1]
    
    print(f"현재가: ${latest['close']:,.2f}")
    print(f"RSI 14: {latest['RSI_14']:.2f}")
    print(f"EMA 20: ${latest['EMA_20']:,.2f}")
    print(f"MACD: {latest['MACD_12_26_9']:.4f}")
    print(f"BB %: {latest['BB_percent_2_0']*100:.1f}%")
    
    # 활성 신호
    print("\n5. 현재 활성 신호")
    signal_columns = [col for col in df_test.columns if any(s in col for s in ['oversold', 'overbought', 'bullish', 'bearish', 'cross'])]
    
    active_signals = []
    for signal in signal_columns:
        if df_test[signal].iloc[-1] == 1:
            active_signals.append(signal)
    
    if active_signals:
        print("🚨 활성 신호:")
        for signal in active_signals:
            print(f"  - {signal}")
    else:
        print("현재 활성 신호 없음")
    
    # 지표 통계
    print("\n6. 지표 통계 (최근 100개 캔들)")
    recent_100 = df_test.tail(100)
    
    print(f"\nRSI 14:")
    print(f"  평균: {recent_100['RSI_14'].mean():.2f}")
    print(f"  최대: {recent_100['RSI_14'].max():.2f}")
    print(f"  최소: {recent_100['RSI_14'].min():.2f}")
    
    # 신호 빈도
    print(f"\n신호 발생 빈도:")
    for signal in signal_columns[:5]:  # 처음 5개만
        count = recent_100[signal].sum()
        print(f"  {signal}: {count}회")

def test_indicator_combinations():
    """지표 조합 테스트"""
    print("\n" + "=" * 60)
    print("지표 조합 생성 테스트")
    print("=" * 60)
    
    combiner = IndicatorCombinations()
    
    # 조합 생성
    print("\n지표 조합 생성 (최대 2개)")
    combinations = combiner.generate_indicator_combinations(max_indicators=2)
    
    print(f"생성된 조합 수: {len(combinations)}")
    
    # 타입별 분류
    type_counts = {}
    for combo in combinations:
        combo_type = combo['type']
        type_counts[combo_type] = type_counts.get(combo_type, 0) + 1
    
    print("\n타입별 조합 수:")
    for combo_type, count in type_counts.items():
        print(f"  {combo_type}: {count}개")
    
    # 샘플 전략
    print("\n샘플 전략 (RSI + EMA):")
    sample_combo = next((c for c in combinations if 'RSI' in c['indicators'] and 'EMA' in c['indicators']), None)
    
    if sample_combo:
        param_combos = combiner.generate_parameter_combinations(sample_combo)
        print(f"파라미터 조합 수: {len(param_combos)}")
        
        # 첫 번째 파라미터 조합으로 전략 생성
        if param_combos:
            strategy = combiner.create_strategy_config(sample_combo, param_combos[0]['parameters'])
            print("\n전략 설정:")
            print(f"  이름: {strategy['name']}")
            print(f"  진입 조건: {strategy['entry_rules']}")
            print(f"  청산 조건: {strategy['exit_rules']}")

if __name__ == "__main__":
    # 1. 지표 계산 테스트
    test_indicator_calculation()
    
    # 2. 지표 조합 테스트
    test_indicator_combinations()
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("스트림릿에서 지표 분석 페이지를 사용해보세요.")
    print("=" * 60)
"""
백테스팅 시스템 통합 테스트
"""
from data_collector import BTCDataCollector
from indicators import TechnicalIndicators
from indicator_combinations import IndicatorCombinations
from backtester import StrategyBacktester
from strategy_optimizer import StrategyOptimizer
from performance_calculator import PerformanceAnalyzer
import pandas as pd
import json
from datetime import datetime

def test_complete_backtesting_workflow():
    """전체 백테스팅 워크플로우 테스트"""
    print("=" * 60)
    print("백테스팅 시스템 통합 테스트")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드")
    collector = BTCDataCollector()
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다. 먼저 데이터를 수집하세요.")
        return
    
    # 최근 3000개 캔들만 사용 (약 4개월)
    df = df.tail(3000).copy()
    print(f"✅ 데이터 로드 완료: {len(df):,} 개의 캔들")
    print(f"기간: {df.index.min()} ~ {df.index.max()}")
    
    # 2. 지표 계산
    print("\n2. 기술적 지표 계산")
    indicators = TechnicalIndicators()
    df = indicators.calculate_all_indicators(df, '1h')
    print(f"✅ 지표 계산 완료: {len(df.columns)} 개의 컬럼")
    
    # 3. 단일 전략 백테스팅
    print("\n3. 단일 전략 백테스팅")
    
    # RSI + 볼린저밴드 전략
    test_strategy = {
        'name': 'RSI_BB_Strategy',
        'type': 'balanced',
        'category': 'momentum_volatility',
        'entry_rules': [
            'RSI_14 < 35',  # 30 -> 35로 완화
            'close < BB_middle_2_0'  # lower -> middle로 완화
        ],
        'exit_rules': [
            'RSI_14 > 65',  # 70 -> 65로 완화
            'close > BB_middle_2_0'  # upper -> middle로 완화
        ]
    }
    
    print(f"전략: {test_strategy['name']}")
    print(f"진입 조건: {test_strategy['entry_rules']}")
    print(f"청산 조건: {test_strategy['exit_rules']}")
    
    # 백테스팅 실행
    backtester = StrategyBacktester(
        initial_capital=10000,
        commission=0.001,
        slippage=0.001
    )
    
    results = backtester.backtest_strategy(df, test_strategy)
    
    print("\n백테스팅 결과:")
    print(f"- 총 거래: {results['total_trades']}")
    print(f"- 승률: {results['win_rate']*100:.1f}%")
    print(f"- 총 수익률: {results['total_return_pct']:.1f}%")
    print(f"- 샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"- 최대 낙폭: {results['max_drawdown']:.1f}%")
    
    # 4. 성과 상세 분석
    print("\n4. 성과 상세 분석")
    analyzer = PerformanceAnalyzer()
    detailed_metrics = analyzer.calculate_performance_metrics(
        backtester.equity_curve,
        results['trades'],
        backtester.initial_capital
    )
    
    print("추가 성과 지표:")
    print(f"- 소르티노 비율: {detailed_metrics['sortino_ratio']:.2f}")
    print(f"- 칼마 비율: {detailed_metrics['calmar_ratio']:.2f}")
    print(f"- 기대값: ${detailed_metrics['expectancy']:.2f}")
    print(f"- Profit Factor: {detailed_metrics['profit_factor']:.2f}")
    
    # 5. 다중 전략 최적화
    print("\n5. 다중 전략 최적화")
    
    # 지표 조합 생성
    combiner = IndicatorCombinations()
    indicator_combos = combiner.generate_indicator_combinations(max_indicators=2)
    
    # 테스트용으로 5개 조합만 선택
    test_combos = indicator_combos[:5]
    print(f"테스트할 지표 조합: {len(test_combos)}개")
    
    # 전략 생성
    test_strategies = []
    for combo in test_combos:
        param_combos = combiner.generate_parameter_combinations(combo)
        # 각 조합당 2개 파라미터만
        for param_combo in param_combos[:2]:
            strategy = combiner.create_strategy_config(combo, param_combo['parameters'])
            test_strategies.append(strategy)
    
    print(f"생성된 전략: {len(test_strategies)}개")
    
    # 최적화 실행
    optimizer = StrategyOptimizer(initial_capital=10000)
    print("\n전략 최적화 실행 중...")
    
    optimization_results = optimizer.optimize_strategies(
        df,
        test_strategies,
        '1h',
        max_workers=1  # Windows 호환성
    )
    
    # 6. 최적화 결과 분석
    print("\n6. 최적화 결과")
    summary = optimization_results['summary_stats']
    
    print(f"\n요약 통계:")
    print(f"- 테스트된 전략: {summary['total_strategies']}")
    print(f"- 평균 수익률: {summary['avg_return']:.1f}%")
    print(f"- 평균 샤프 비율: {summary['avg_sharpe']:.2f}")
    print(f"- 수익 전략 비율: {summary['profitable_ratio']*100:.1f}%")
    
    # 최고 전략
    if optimization_results['best_strategies']['overall_best']:
        print("\n🏆 최고 성과 전략 (상위 3개):")
        for i, strategy in enumerate(optimization_results['best_strategies']['overall_best'][:3]):
            print(f"\n{i+1}. {strategy['name']}")
            print(f"   - 수익률: {strategy['performance']['total_return_pct']:.1f}%")
            print(f"   - 샤프 비율: {strategy['performance']['sharpe_ratio']:.2f}")
            print(f"   - 승률: {strategy['performance']['win_rate']*100:.0f}%")
            print(f"   - 최대 낙폭: {strategy['performance']['max_drawdown']:.1f}%")
            print(f"   - 거래 수: {strategy['performance']['total_trades']}")
    
    # 7. 결과 저장
    print("\n7. 결과 저장")
    
    # 단순화된 결과 (저장용)
    save_results = {
        'test_date': datetime.now().isoformat(),
        'data_period': f"{df.index.min()} ~ {df.index.max()}",
        'single_strategy_test': {
            'strategy_name': test_strategy['name'],
            'performance': {
                'total_return_pct': results['total_return_pct'],
                'sharpe_ratio': results['sharpe_ratio'],
                'win_rate': results['win_rate'],
                'max_drawdown': results['max_drawdown']
            }
        },
        'optimization_summary': summary,
        'top_3_strategies': []
    }
    
    # 상위 3개 전략
    if optimization_results['best_strategies']['overall_best']:
        for strategy in optimization_results['best_strategies']['overall_best'][:3]:
            save_results['top_3_strategies'].append({
                'name': strategy['name'],
                'performance': strategy['performance']
            })
    
    # JSON 파일로 저장
    import os
    filepath = os.path.join('results', 'backtesting_test_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 결과 저장 완료: {filepath}")
    
    # 8. 최적 전략의 거래 신호 시각화 데이터
    if optimization_results['best_strategies']['overall_best']:
        best_strategy = optimization_results['best_strategies']['overall_best'][0]
        print(f"\n8. 최적 전략 거래 신호 분석")
        print(f"전략: {best_strategy['name']}")
        
        # 해당 전략으로 다시 백테스팅 (거래 신호 확인용)
        backtester = StrategyBacktester(initial_capital=10000)
        best_results = backtester.backtest_strategy(df, best_strategy['strategy'])
        
        if best_results['trades']:
            print(f"\n최근 5개 거래:")
            for trade in best_results['trades'][-5:]:
                print(f"- {trade['entry_time']} → {trade['exit_time']}")
                print(f"  수익: ${trade['profit']:.2f} ({trade['profit_pct']:.1f}%)")

def test_performance_metrics():
    """성과 지표 계산 테스트"""
    print("\n" + "=" * 60)
    print("성과 지표 상세 테스트")
    print("=" * 60)
    
    # 간단한 백테스팅
    collector = BTCDataCollector()
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return
    
    df = df.tail(1000).copy()
    
    # 지표 계산
    indicators = TechnicalIndicators()
    df = indicators.calculate_all_indicators(df, '1h')
    
    # MACD 전략
    strategy = {
        'name': 'MACD_Strategy',
        'type': 'trend',
        'entry_rules': ['MACD_12_26_9 > MACD_signal_12_26_9'],
        'exit_rules': ['MACD_12_26_9 < MACD_signal_12_26_9']
    }
    
    # 백테스팅
    backtester = StrategyBacktester()
    results = backtester.backtest_strategy(df, strategy)
    
    # 상세 성과 분석
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_performance_metrics(
        backtester.equity_curve,
        results['trades'],
        backtester.initial_capital
    )
    
    # 성과 보고서
    report = analyzer.generate_performance_report(metrics)
    print(report)
    
    # 시간대별 분석
    if 'time_analysis' in metrics and metrics['time_analysis']:
        print("\n시간대별 성과:")
        hourly = metrics['time_analysis'].get('hourly_performance', {})
        for hour in sorted(hourly.keys())[:5]:  # 처음 5개 시간대
            stats = hourly[hour]
            print(f"  {hour}시: 거래 {stats['trades']}건, 승률 {stats['win_rate']*100:.0f}%")

if __name__ == "__main__":
    # 1. 전체 워크플로우 테스트
    test_complete_backtesting_workflow()
    
    # 2. 성과 지표 상세 테스트
    test_performance_metrics()
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("스트림릿에서 백테스팅 기능을 사용해보세요.")
    print("=" * 60)
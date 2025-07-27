"""
전략 최적화 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import os
from datetime import datetime

import config
import utils
from backtester import StrategyBacktester
from indicator_combinations import IndicatorCombinations
from indicators import TechnicalIndicators

logger = utils.setup_logger(__name__)

class StrategyOptimizer:
    """전략 최적화 클래스"""
    
    def __init__(self, initial_capital: float = 10000):
        """초기화"""
        self.initial_capital = initial_capital
        self.results = []
        self.best_strategies = {}
        
    def optimize_strategies(self, df: pd.DataFrame, strategies: List[Dict], 
                          timeframe: str, max_workers: int = 4) -> Dict:
        """여러 전략 최적화
        
        Args:
            df: 가격 및 지표 데이터
            strategies: 전략 리스트
            timeframe: 타임프레임
            max_workers: 병렬 처리 워커 수
        
        Returns:
            최적화 결과
        """
        logger.info(f"전략 최적화 시작: {len(strategies)}개 전략, {timeframe}")
        
        # 지표가 계산되지 않은 경우 계산
        if 'RSI_14' not in df.columns:
            logger.info("지표 계산 중...")
            indicators = TechnicalIndicators()
            df = indicators.calculate_all_indicators(df, timeframe)
        
        # Windows에서는 순차 처리, 그 외에는 병렬 처리
        self.results = []
        
        import platform
        if platform.system() == 'Windows' or max_workers == 1:
            # 순차 처리
            with tqdm(total=len(strategies), desc="백테스팅 진행") as pbar:
                for strategy in strategies:
                    try:
                        result = self._backtest_single_strategy(df, strategy)
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"백테스팅 실패 ({strategy['name']}): {e}")
                    pbar.update(1)
        else:
            # 병렬 처리 (Linux, Mac)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 백테스팅 작업 제출
                future_to_strategy = {
                    executor.submit(self._backtest_single_strategy, df, strategy): strategy 
                    for strategy in strategies
                }
                
                # 진행 상황 표시
                with tqdm(total=len(strategies), desc="백테스팅 진행") as pbar:
                    for future in as_completed(future_to_strategy):
                        strategy = future_to_strategy[future]
                        try:
                            result = future.result()
                            if result:
                                self.results.append(result)
                        except Exception as e:
                            logger.error(f"백테스팅 실패 ({strategy['name']}): {e}")
                        pbar.update(1)
        
        # 결과 분석
        optimization_results = self._analyze_results(timeframe)
        
        logger.info(f"전략 최적화 완료: {len(self.results)}개 성공")
        
        return optimization_results
    
    def _backtest_single_strategy(self, df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
        """단일 전략 백테스팅"""
        try:
            backtester = StrategyBacktester(
                initial_capital=self.initial_capital,
                commission=config.BACKTEST_CONFIG['commission'],
                slippage=config.BACKTEST_CONFIG['slippage']
            )
            
            result = backtester.backtest_strategy(df, strategy)
            result['strategy'] = strategy
            
            return result
        except Exception as e:
            logger.error(f"백테스팅 오류 ({strategy['name']}): {e}")
            return None
    
    def _analyze_results(self, timeframe: str) -> Dict:
        """결과 분석 및 최적 전략 선택"""
        if not self.results:
            return {
                'timeframe': timeframe,
                'total_strategies': 0,
                'best_strategies': {}
            }
        
        # DataFrame으로 변환
        results_df = pd.DataFrame(self.results)
        
        # 주요 지표별 정렬
        metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate', 'profit_factor']
        
        best_strategies = {}
        
        for metric in metrics:
            # 해당 지표로 정렬 (내림차순)
            sorted_df = results_df.sort_values(metric, ascending=False)
            
            # 상위 5개 전략
            top_strategies = []
            for _, row in sorted_df.head(5).iterrows():
                strategy_info = {
                    'name': row['strategy_name'],
                    'type': row['strategy_type'],
                    'performance': {
                        'total_return_pct': row['total_return_pct'],
                        'sharpe_ratio': row['sharpe_ratio'],
                        'win_rate': row['win_rate'],
                        'max_drawdown': row['max_drawdown'],
                        'total_trades': row['total_trades'],
                        'profit_factor': row['profit_factor']
                    },
                    'strategy': row['strategy']
                }
                top_strategies.append(strategy_info)
            
            best_strategies[f'best_by_{metric}'] = top_strategies
        
        # 종합 점수 계산 (모든 지표 고려)
        results_df['composite_score'] = (
            results_df['total_return_pct'].rank(pct=True) * 0.3 +
            results_df['sharpe_ratio'].rank(pct=True) * 0.3 +
            results_df['win_rate'].rank(pct=True) * 0.2 +
            results_df['profit_factor'].rank(pct=True) * 0.2
        )
        
        # 종합 점수 기준 최고 전략
        sorted_df = results_df.sort_values('composite_score', ascending=False)
        
        overall_best = []
        for _, row in sorted_df.head(10).iterrows():
            strategy_info = {
                'name': row['strategy_name'],
                'type': row['strategy_type'],
                'composite_score': row['composite_score'],
                'performance': {
                    'total_return_pct': row['total_return_pct'],
                    'sharpe_ratio': row['sharpe_ratio'],
                    'win_rate': row['win_rate'],
                    'max_drawdown': row['max_drawdown'],
                    'total_trades': row['total_trades'],
                    'profit_factor': row['profit_factor']
                },
                'strategy': row['strategy']
            }
            overall_best.append(strategy_info)
        
        best_strategies['overall_best'] = overall_best
        
        # 전략 타입별 최고
        type_best = {}
        for strategy_type in results_df['strategy_type'].unique():
            type_df = results_df[results_df['strategy_type'] == strategy_type]
            if not type_df.empty:
                best_row = type_df.loc[type_df['composite_score'].idxmax()]
                type_best[strategy_type] = {
                    'name': best_row['strategy_name'],
                    'performance': {
                        'total_return_pct': best_row['total_return_pct'],
                        'sharpe_ratio': best_row['sharpe_ratio'],
                        'win_rate': best_row['win_rate'],
                        'max_drawdown': best_row['max_drawdown'],
                        'total_trades': best_row['total_trades']
                    }
                }
        
        best_strategies['best_by_type'] = type_best
        
        # 통계 요약
        summary_stats = {
            'total_strategies': len(results_df),
            'avg_return': results_df['total_return_pct'].mean(),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'avg_win_rate': results_df['win_rate'].mean(),
            'profitable_strategies': len(results_df[results_df['total_return_pct'] > 0]),
            'profitable_ratio': len(results_df[results_df['total_return_pct'] > 0]) / len(results_df)
        }
        
        return {
            'timeframe': timeframe,
            'optimization_time': datetime.now().isoformat(),
            'summary_stats': summary_stats,
            'best_strategies': best_strategies,
            'all_results': self.results
        }
    
    def save_optimization_results(self, results: Dict, filename: str = None):
        """최적화 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_results_{results['timeframe']}_{timestamp}.json"
        
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        # 거래 리스트가 너무 크면 제거 (요약만 저장)
        save_results = results.copy()
        if 'all_results' in save_results:
            for result in save_results['all_results']:
                if 'trades' in result:
                    result['trades_count'] = len(result['trades'])
                    result['trades'] = result['trades'][:10]  # 처음 10개만 저장
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"최적화 결과 저장: {filepath}")
        return filepath
    
    def find_optimal_parameters(self, df: pd.DataFrame, indicator_combo: Dict, 
                              timeframe: str) -> Dict:
        """특정 지표 조합의 최적 파라미터 찾기"""
        logger.info(f"파라미터 최적화: {indicator_combo['name']}")
        
        # 파라미터 조합 생성
        combiner = IndicatorCombinations()
        param_combinations = combiner.generate_parameter_combinations(indicator_combo)
        
        # 각 파라미터 조합에 대한 전략 생성
        strategies = []
        for param_combo in param_combinations:
            strategy = combiner.create_strategy_config(indicator_combo, param_combo['parameters'])
            strategies.append(strategy)
        
        # 최적화 실행
        optimization_results = self.optimize_strategies(df, strategies, timeframe)
        
        # 최적 파라미터 추출
        if optimization_results['best_strategies']['overall_best']:
            best_strategy = optimization_results['best_strategies']['overall_best'][0]
            return {
                'indicator_combination': indicator_combo,
                'optimal_parameters': best_strategy['strategy']['indicators'],
                'performance': best_strategy['performance'],
                'strategy_config': best_strategy['strategy']
            }
        
        return None

# 테스트 함수
def test_optimizer():
    """전략 최적화 테스트"""
    from data_collector import BTCDataCollector
    
    print("=" * 60)
    print("전략 최적화 테스트")
    print("=" * 60)
    
    # 데이터 로드
    collector = BTCDataCollector()
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return
    
    # 최근 2000개 캔들만 사용
    df = df.tail(2000).copy()
    
    print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
    print(f"캔들 수: {len(df)}")
    
    # 지표 조합 생성
    combiner = IndicatorCombinations()
    indicator_combos = combiner.generate_indicator_combinations(max_indicators=2)
    
    # 처음 5개 조합만 테스트
    test_combos = indicator_combos[:5]
    
    # 전략 생성
    all_strategies = []
    for combo in test_combos:
        param_combos = combiner.generate_parameter_combinations(combo)
        for param_combo in param_combos[:3]:  # 각 조합당 3개 파라미터만
            strategy = combiner.create_strategy_config(combo, param_combo['parameters'])
            all_strategies.append(strategy)
    
    print(f"\n테스트할 전략 수: {len(all_strategies)}")
    
    # 최적화 실행
    optimizer = StrategyOptimizer(initial_capital=10000)
    
    print("\n최적화 실행 중...")
    results = optimizer.optimize_strategies(df, all_strategies, '1h', max_workers=1)
    
    # 결과 출력
    print("\n📊 최적화 결과:")
    print(f"테스트된 전략 수: {results['summary_stats']['total_strategies']}")
    print(f"평균 수익률: {results['summary_stats']['avg_return']:.1f}%")
    print(f"평균 샤프 비율: {results['summary_stats']['avg_sharpe']:.2f}")
    print(f"수익 전략 비율: {results['summary_stats']['profitable_ratio']*100:.1f}%")
    
    # 최고 전략
    if results['best_strategies']['overall_best']:
        print("\n🏆 최고 전략 (종합 점수):")
        best = results['best_strategies']['overall_best'][0]
        print(f"전략명: {best['name']}")
        print(f"수익률: {best['performance']['total_return_pct']:.1f}%")
        print(f"샤프 비율: {best['performance']['sharpe_ratio']:.2f}")
        print(f"승률: {best['performance']['win_rate']*100:.1f}%")
        print(f"최대 낙폭: {best['performance']['max_drawdown']:.1f}%")
        print(f"거래 수: {best['performance']['total_trades']}")
    
    # 결과 저장
    filepath = optimizer.save_optimization_results(results)
    print(f"\n결과 저장: {filepath}")

if __name__ == "__main__":
    test_optimizer()
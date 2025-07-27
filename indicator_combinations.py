"""
지표 조합 생성 및 관리 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from itertools import combinations, product
import json
import os

import config
import utils

logger = utils.setup_logger(__name__)

class IndicatorCombinations:
    """지표 조합 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        self.base_indicators = {
            'momentum': ['RSI', 'STOCH', 'CCI', 'WILLR', 'ROC'],
            'trend': ['EMA', 'SMA', 'MACD', 'ADX'],
            'volatility': ['BB', 'ATR'],
            'volume': ['OBV', 'Volume_SMA', 'Volume_ROC']
        }
        
        self.combinations = []
        
    def generate_indicator_combinations(self, max_indicators: int = 3) -> List[Dict]:
        """지표 조합 생성"""
        logger.info(f"지표 조합 생성 시작 (최대 {max_indicators}개)")
        
        all_combinations = []
        
        # 1. 단일 카테고리 조합
        for category, indicators in self.base_indicators.items():
            # 각 카테고리에서 1-3개 선택
            for r in range(1, min(len(indicators), max_indicators) + 1):
                for combo in combinations(indicators, r):
                    all_combinations.append({
                        'indicators': list(combo),
                        'type': 'single_category',
                        'category': category,
                        'name': f"{category}_{'_'.join(combo)}"
                    })
        
        # 2. 다중 카테고리 조합 (균형잡힌 전략)
        # 모멘텀 + 트렌드
        for mom in self.base_indicators['momentum'][:3]:
            for trend in self.base_indicators['trend'][:3]:
                all_combinations.append({
                    'indicators': [mom, trend],
                    'type': 'balanced',
                    'category': 'momentum_trend',
                    'name': f"balanced_{mom}_{trend}"
                })
        
        # 트렌드 + 변동성
        for trend in self.base_indicators['trend'][:3]:
            for vol in self.base_indicators['volatility']:
                all_combinations.append({
                    'indicators': [trend, vol],
                    'type': 'balanced',
                    'category': 'trend_volatility',
                    'name': f"balanced_{trend}_{vol}"
                })
        
        # 3. 전체 조합 (모든 카테고리에서 하나씩)
        if max_indicators >= 4:
            all_combinations.append({
                'indicators': ['RSI', 'EMA', 'BB', 'OBV'],
                'type': 'comprehensive',
                'category': 'all',
                'name': 'comprehensive_basic'
            })
            
            all_combinations.append({
                'indicators': ['STOCH', 'MACD', 'ATR', 'Volume_SMA'],
                'type': 'comprehensive',
                'category': 'all',
                'name': 'comprehensive_advanced'
            })
        
        self.combinations = all_combinations
        logger.info(f"총 {len(all_combinations)}개의 지표 조합 생성 완료")
        
        return all_combinations
    
    def generate_parameter_combinations(self, indicator_combo: Dict) -> List[Dict]:
        """특정 지표 조합에 대한 파라미터 조합 생성"""
        param_combinations = []
        
        # 각 지표별 파라미터 설정
        param_space = {
            'RSI': [9, 14, 21],
            'EMA': [10, 20, 50],
            'SMA': [10, 20, 50],
            'MACD': [(12, 26, 9), (8, 21, 5)],
            'STOCH': [(14, 3, 3), (5, 3, 3)],
            'BB': [(20, 1.5), (20, 2.0), (20, 2.5)],
            'ATR': [14, 21],
            'ADX': [14, 21],
            'CCI': [14, 20],
            'WILLR': [14, 21],
            'ROC': [10, 20],
            'OBV': [None],  # 파라미터 없음
            'Volume_SMA': [20],
            'Volume_ROC': [10]
        }
        
        # 지표별 파라미터 추출
        indicator_params = []
        for indicator in indicator_combo['indicators']:
            if indicator in param_space:
                indicator_params.append(param_space[indicator])
            else:
                indicator_params.append([None])
        
        # 모든 파라미터 조합 생성
        for params in product(*indicator_params):
            param_dict = {}
            for i, indicator in enumerate(indicator_combo['indicators']):
                param_dict[indicator] = params[i]
            
            param_combinations.append({
                'combination_name': indicator_combo['name'],
                'indicators': indicator_combo['indicators'],
                'parameters': param_dict,
                'param_string': self._generate_param_string(param_dict)
            })
        
        return param_combinations
    
    def _generate_param_string(self, param_dict: Dict) -> str:
        """파라미터를 문자열로 변환"""
        parts = []
        for indicator, param in param_dict.items():
            if param is None:
                parts.append(indicator)
            elif isinstance(param, tuple):
                parts.append(f"{indicator}{'_'.join(map(str, param))}")
            else:
                parts.append(f"{indicator}{param}")
        return "_".join(parts)
    
    def create_strategy_config(self, indicator_combo: Dict, parameters: Dict) -> Dict:
        """전략 설정 생성"""
        strategy = {
            'name': f"{indicator_combo['name']}_{self._generate_param_string(parameters)}",
            'type': indicator_combo['type'],
            'category': indicator_combo['category'],
            'indicators': {},
            'entry_rules': [],
            'exit_rules': []
        }
        
        # 각 지표별 설정
        for indicator in indicator_combo['indicators']:
            param = parameters.get(indicator)
            
            if indicator == 'RSI':
                strategy['indicators'][f'RSI_{param}'] = {
                    'type': 'momentum',
                    'period': param,
                    'overbought': 70,
                    'oversold': 30
                }
                strategy['entry_rules'].append(f"RSI_{param} < 35")  # 30 -> 35
                strategy['exit_rules'].append(f"RSI_{param} > 65")  # 70 -> 65
                
            elif indicator == 'EMA':
                strategy['indicators'][f'EMA_{param}'] = {
                    'type': 'trend',
                    'period': param
                }
                strategy['entry_rules'].append(f"close > EMA_{param}")
                
            elif indicator == 'MACD':
                fast, slow, signal = param
                strategy['indicators'][f'MACD_{fast}_{slow}_{signal}'] = {
                    'type': 'trend',
                    'fast': fast,
                    'slow': slow,
                    'signal': signal
                }
                strategy['entry_rules'].append(f"MACD_{fast}_{slow}_{signal} > MACD_signal_{fast}_{slow}_{signal}")
                
            elif indicator == 'BB':
                period, std = param
                std_str = str(std).replace('.', '_')
                strategy['indicators'][f'BB_{period}_{std_str}'] = {
                    'type': 'volatility',
                    'period': period,
                    'std_dev': std
                }
                strategy['entry_rules'].append(f"close < BB_middle_{std_str}")  # lower -> middle
                strategy['exit_rules'].append(f"close > BB_middle_{std_str}")  # upper -> middle
                
            elif indicator == 'STOCH':
                k, d, smooth = param
                strategy['indicators'][f'STOCH_{k}_{d}_{smooth}'] = {
                    'type': 'momentum',
                    'k_period': k,
                    'd_period': d,
                    'smooth': smooth
                }
                strategy['entry_rules'].append(f"STOCH_K_{k}_{d}_{smooth} < 30")  # 20 -> 30
                strategy['exit_rules'].append(f"STOCH_K_{k}_{d}_{smooth} > 70")  # 80 -> 70
        
        return strategy
    
    def get_all_strategy_configs(self, max_indicators: int = 3) -> List[Dict]:
        """모든 전략 설정 생성"""
        all_strategies = []
        
        # 지표 조합 생성
        indicator_combos = self.generate_indicator_combinations(max_indicators)
        
        # 각 조합에 대한 파라미터 조합 생성
        for combo in indicator_combos:
            param_combos = self.generate_parameter_combinations(combo)
            
            # 각 파라미터 조합에 대한 전략 생성
            for param_combo in param_combos:
                strategy = self.create_strategy_config(combo, param_combo['parameters'])
                all_strategies.append(strategy)
        
        logger.info(f"총 {len(all_strategies)}개의 전략 설정 생성 완료")
        return all_strategies
    
    def save_strategy_configs(self, strategies: List[Dict], filename: str = 'strategy_configs.json'):
        """전략 설정 저장"""
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
        
        logger.info(f"전략 설정 저장 완료: {filepath}")
        return filepath
    
    def load_strategy_configs(self, filename: str = 'strategy_configs.json') -> List[Dict]:
        """전략 설정 로드"""
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"전략 설정 파일이 없습니다: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
        
        logger.info(f"전략 설정 로드 완료: {len(strategies)}개")
        return strategies
    
    def filter_strategies(self, strategies: List[Dict], 
                         indicator_type: str = None,
                         min_indicators: int = None,
                         max_indicators: int = None) -> List[Dict]:
        """전략 필터링"""
        filtered = strategies
        
        if indicator_type:
            filtered = [s for s in filtered if s['type'] == indicator_type]
        
        if min_indicators:
            filtered = [s for s in filtered if len(s['indicators']) >= min_indicators]
        
        if max_indicators:
            filtered = [s for s in filtered if len(s['indicators']) <= max_indicators]
        
        return filtered

# 테스트 함수
def test_combinations():
    """지표 조합 테스트"""
    import os
    
    print("=" * 60)
    print("지표 조합 생성 테스트")
    print("=" * 60)
    
    combiner = IndicatorCombinations()
    
    # 1. 지표 조합 생성
    print("\n1. 지표 조합 생성")
    indicator_combos = combiner.generate_indicator_combinations(max_indicators=2)
    
    print(f"생성된 조합 수: {len(indicator_combos)}")
    print("\n샘플 조합 (처음 5개):")
    for i, combo in enumerate(indicator_combos[:5]):
        print(f"  {i+1}. {combo['name']}: {combo['indicators']}")
    
    # 2. 파라미터 조합 생성
    print("\n2. 파라미터 조합 생성")
    sample_combo = indicator_combos[0]
    param_combos = combiner.generate_parameter_combinations(sample_combo)
    
    print(f"'{sample_combo['name']}'에 대한 파라미터 조합 수: {len(param_combos)}")
    print("\n샘플 파라미터 조합 (처음 3개):")
    for i, param in enumerate(param_combos[:3]):
        print(f"  {i+1}. {param['param_string']}")
        print(f"     파라미터: {param['parameters']}")
    
    # 3. 전략 설정 생성
    print("\n3. 전략 설정 생성")
    all_strategies = combiner.get_all_strategy_configs(max_indicators=2)
    
    print(f"총 전략 수: {len(all_strategies)}")
    
    # 타입별 전략 수
    type_counts = {}
    for strategy in all_strategies:
        strategy_type = strategy['type']
        type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1
    
    print("\n타입별 전략 수:")
    for strategy_type, count in type_counts.items():
        print(f"  {strategy_type}: {count}개")
    
    # 4. 전략 저장
    print("\n4. 전략 설정 저장")
    filepath = combiner.save_strategy_configs(all_strategies[:100])  # 처음 100개만 저장
    print(f"저장 완료: {filepath}")
    
    # 샘플 전략 출력
    print("\n샘플 전략 설정:")
    sample_strategy = all_strategies[0]
    print(json.dumps(sample_strategy, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_combinations()
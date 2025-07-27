"""
백테스팅 엔진 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os

import config
import utils
from indicators import TechnicalIndicators

logger = utils.setup_logger(__name__)

@dataclass
class Trade:
    """거래 정보 클래스"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    position_size: float = 1.0
    trade_type: str = 'long'  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def profit(self) -> float:
        if not self.exit_price:
            return 0.0
        
        if self.trade_type == 'long':
            return (self.exit_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - self.exit_price) * self.position_size
    
    @property
    def profit_percentage(self) -> float:
        if not self.exit_price:
            return 0.0
        
        if self.trade_type == 'long':
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def duration(self) -> pd.Timedelta:
        if not self.exit_time:
            return pd.Timedelta(0)
        return self.exit_time - self.entry_time

class StrategyBacktester:
    """전략 백테스팅 클래스"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, slippage: float = 0.001):
        """초기화
        
        Args:
            initial_capital: 초기 자본금
            commission: 수수료율 (0.1% = 0.001)
            slippage: 슬리피지율 (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades: List[Trade] = []
        self.capital = initial_capital
        self.positions = 0
        self.equity_curve = []
        
    def _calculate_position_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> float:
        """포지션 크기 계산
        
        Args:
            capital: 현재 자본금
            price: 현재 가격
            risk_per_trade: 거래당 리스크 (2% = 0.02)
        """
        # 간단한 고정 비율 방식
        position_value = capital * risk_per_trade
        position_size = position_value / price
        return position_size
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """슬리피지 적용"""
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def _calculate_commission(self, value: float) -> float:
        """수수료 계산"""
        return value * self.commission
    
    def backtest_strategy(self, df: pd.DataFrame, strategy: Dict) -> Dict:
        """전략 백테스팅 실행
        
        Args:
            df: 가격 및 지표 데이터
            strategy: 전략 설정
        
        Returns:
            백테스팅 결과
        """
        logger.info(f"백테스팅 시작: {strategy['name']}")
        logger.debug(f"진입 조건: {strategy['entry_rules']}")
        logger.debug(f"청산 조건: {strategy['exit_rules']}")
        
        # 초기화
        self.trades = []
        self.capital = self.initial_capital
        self.equity_curve = []
        
        current_trade = None
        entry_signals_count = 0
        exit_signals_count = 0
        
        # 필요한 컬럼 확인 및 첫 유효 인덱스 찾기
        all_columns = set()
        for rule in strategy.get('entry_rules', []) + strategy.get('exit_rules', []):
            for word in rule.split():
                if word in df.columns:
                    all_columns.add(word)
        
        # 모든 필요한 컬럼에서 NaN이 아닌 첫 인덱스 찾기
        first_valid_idx = 0
        if all_columns:
            for col in all_columns:
                if col in df.columns:
                    first_valid = df[col].first_valid_index()
                    if first_valid is not None:
                        first_valid_idx = max(first_valid_idx, df.index.get_loc(first_valid))
        
        # 시작 인덱스 설정
        start_idx = max(first_valid_idx + 1, 50)  # 최소 50개 캔들 이후부터 시작
        
        logger.info(f"백테스팅 시작 인덱스: {start_idx}/{len(df)}")
        
        # 각 캔들에 대해 반복
        for i in range(start_idx, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            
            # 진입 신호 확인
            if current_trade is None:
                entry_signal = self._check_entry_signal(df.iloc[:i+1], strategy['entry_rules'])
                
                if entry_signal:
                    entry_signals_count += 1
                    # 포지션 크기 계산
                    position_size = self._calculate_position_size(self.capital, current_price)
                    
                    # 진입 가격 (슬리피지 적용)
                    entry_price = self._apply_slippage(current_price, is_buy=True)
                    
                    # 수수료 차감
                    commission = self._calculate_commission(entry_price * position_size)
                    self.capital -= commission
                    
                    # 거래 생성
                    current_trade = Trade(
                        entry_time=current_time,
                        entry_price=entry_price,
                        position_size=position_size,
                        trade_type='long'
                    )
                    
                    logger.debug(f"진입: {current_time} @ ${entry_price:.2f}")
            
            # 청산 신호 확인
            elif current_trade and current_trade.is_open:
                exit_signal = self._check_exit_signal(df.iloc[:i+1], strategy['exit_rules'])
                
                if exit_signal:
                    exit_signals_count += 1
                
                if exit_signal or i == len(df) - 1:  # 마지막 캔들에서는 강제 청산
                    # 청산 가격 (슬리피지 적용)
                    exit_price = self._apply_slippage(current_price, is_buy=False)
                    
                    # 거래 완료
                    current_trade.exit_time = current_time
                    current_trade.exit_price = exit_price
                    
                    # 수익 계산
                    profit = current_trade.profit
                    
                    # 수수료 차감
                    commission = self._calculate_commission(exit_price * current_trade.position_size)
                    profit -= commission
                    
                    # 자본금 업데이트
                    self.capital += profit
                    
                    # 거래 기록
                    self.trades.append(current_trade)
                    current_trade = None
                    
                    logger.debug(f"청산: {current_time} @ ${exit_price:.2f}, 수익: ${profit:.2f}")
            
            # 자산 곡선 기록
            current_equity = self.capital
            if current_trade and current_trade.is_open:
                # 미실현 손익 포함
                unrealized_pnl = (current_price - current_trade.entry_price) * current_trade.position_size
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity,
                'capital': self.capital
            })
        
        # 백테스팅 결과 계산
        results = self._calculate_results(df)
        results['strategy_name'] = strategy['name']
        results['strategy_type'] = strategy.get('type', 'unknown')
        results['entry_signals'] = entry_signals_count
        results['exit_signals'] = exit_signals_count
        
        logger.info(f"백테스팅 완료: {len(self.trades)} 거래, 최종 자본: ${self.capital:.2f}")
        logger.info(f"진입 신호: {entry_signals_count}회, 청산 신호: {exit_signals_count}회")
        
        return results
    
    def _check_entry_signal(self, df: pd.DataFrame, entry_rules: List[str]) -> bool:
        """진입 신호 확인"""
        if not entry_rules:
            return False
        
        latest = df.iloc[-1]
        
        for rule in entry_rules:
            try:
                # 규칙 파싱 및 평가
                if '<' in rule:
                    parts = rule.split('<')
                    left_value = self._get_value(latest, parts[0].strip())
                    right_value = self._get_value(latest, parts[1].strip())
                    
                    # NaN 체크
                    if pd.isna(left_value) or pd.isna(right_value):
                        return False
                    
                    if not (left_value < right_value):
                        return False
                elif '>' in rule:
                    parts = rule.split('>')
                    left_value = self._get_value(latest, parts[0].strip())
                    right_value = self._get_value(latest, parts[1].strip())
                    
                    # NaN 체크
                    if pd.isna(left_value) or pd.isna(right_value):
                        return False
                    
                    if not (left_value > right_value):
                        return False
            except Exception as e:
                logger.debug(f"진입 규칙 평가 실패: {rule}, 오류: {e}")
                return False
        
        return True
    
    def _check_exit_signal(self, df: pd.DataFrame, exit_rules: List[str]) -> bool:
        """청산 신호 확인"""
        if not exit_rules:
            return False
        
        latest = df.iloc[-1]
        
        for rule in exit_rules:
            try:
                # 규칙 파싱 및 평가
                if '<' in rule:
                    parts = rule.split('<')
                    left_value = self._get_value(latest, parts[0].strip())
                    right_value = self._get_value(latest, parts[1].strip())
                    
                    # NaN 체크
                    if pd.isna(left_value) or pd.isna(right_value):
                        return False
                    
                    if left_value < right_value:
                        return True
                elif '>' in rule:
                    parts = rule.split('>')
                    left_value = self._get_value(latest, parts[0].strip())
                    right_value = self._get_value(latest, parts[1].strip())
                    
                    # NaN 체크
                    if pd.isna(left_value) or pd.isna(right_value):
                        return False
                    
                    if left_value > right_value:
                        return True
            except Exception as e:
                logger.debug(f"청산 규칙 평가 실패: {rule}, 오류: {e}")
                continue
        
        return False
    
    def _get_value(self, data: pd.Series, column_or_value: str) -> float:
        """컬럼명 또는 숫자 값 가져오기"""
        try:
            # 숫자인 경우
            return float(column_or_value)
        except ValueError:
            # 컬럼명인 경우
            if column_or_value in data.index:
                value = data[column_or_value]
                if pd.isna(value):
                    raise ValueError(f"값이 NaN입니다: {column_or_value}")
                return float(value)
            else:
                raise ValueError(f"컬럼을 찾을 수 없음: {column_or_value}")
    
    def _calculate_results(self, df: pd.DataFrame) -> Dict:
        """백테스팅 결과 계산"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_trade_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'final_capital': self.capital,  # 추가
                'trades': []
            }
        
        # 거래 통계
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # 수익률
        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # 평균 수익/손실
        avg_win = np.mean([t.profit for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.profit for t in losing_trades]) if losing_trades else 0
        avg_trade_return = np.mean([t.profit for t in self.trades])
        
        # Profit Factor
        gross_profit = sum([t.profit for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t.profit for t in losing_trades])) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 자산 곡선 분석
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 일간 수익률
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # 샤프 비율 (연율화)
        returns_mean = equity_df['returns'].mean()
        returns_std = equity_df['returns'].std()
        sharpe_ratio = np.sqrt(252) * returns_mean / returns_std if returns_std > 0 else 0
        
        # 최대 낙폭
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # 거래 리스트
        trades_list = []
        for t in self.trades:
            trades_list.append({
                'entry_time': t.entry_time.strftime('%Y-%m-%d %H:%M'),
                'exit_time': t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else None,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'profit': t.profit,
                'profit_pct': t.profit_percentage,
                'duration': str(t.duration) if t.exit_time else None
            })
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_return': avg_trade_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.capital,
            'trades': trades_list
        }

# 테스트 함수
def test_backtester():
    """백테스터 테스트"""
    from data_collector import BTCDataCollector
    from indicators import TechnicalIndicators
    
    print("=" * 60)
    print("백테스팅 엔진 테스트")
    print("=" * 60)
    
    # 데이터 로드
    collector = BTCDataCollector()
    df = collector.load_data('1h')
    
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return
    
    # 최근 1000개 캔들만 사용
    df = df.tail(1000).copy()
    
    # 지표 계산
    indicators = TechnicalIndicators()
    df = indicators.calculate_all_indicators(df, '1h')
    
    # 샘플 전략
    sample_strategy = {
        'name': 'RSI_BB_Strategy',
        'type': 'balanced',
        'entry_rules': ['RSI_14 < 35', 'close < BB_middle_2_0'],
        'exit_rules': ['RSI_14 > 65', 'close > BB_middle_2_0']
    }
    
    # 백테스팅 실행
    backtester = StrategyBacktester(
        initial_capital=10000,
        commission=0.001,
        slippage=0.001
    )
    
    print(f"\n전략: {sample_strategy['name']}")
    print(f"진입 조건: {sample_strategy['entry_rules']}")
    print(f"청산 조건: {sample_strategy['exit_rules']}")
    print("\n백테스팅 실행 중...")
    
    results = backtester.backtest_strategy(df, sample_strategy)
    
    # 결과 출력
    print("\n📊 백테스팅 결과:")
    print(f"총 거래 수: {results['total_trades']}")
    print(f"승리 거래: {results['winning_trades']}")
    print(f"패배 거래: {results['losing_trades']}")
    print(f"승률: {results['win_rate']*100:.1f}%")
    print(f"총 수익: ${results['total_return']:,.2f} ({results['total_return_pct']:.1f}%)")
    print(f"샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"최대 낙폭: {results['max_drawdown']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"최종 자본: ${results['final_capital']:,.2f}")
    
    # 최근 거래 5개
    if results['trades']:
        print("\n최근 거래 (최대 5개):")
        for i, trade in enumerate(results['trades'][-5:]):
            print(f"{i+1}. {trade['entry_time']} -> {trade['exit_time']}")
            print(f"   수익: ${trade['profit']:.2f} ({trade['profit_pct']:.1f}%)")

if __name__ == "__main__":
    test_backtester()
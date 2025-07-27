"""
성과 계산 및 분석 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import config
import utils

logger = utils.setup_logger(__name__)

class PerformanceAnalyzer:
    """성과 분석 클래스"""
    
    def __init__(self):
        """초기화"""
        self.metrics = {}
        
    def calculate_performance_metrics(self, equity_curve: List[Dict], trades: List[Dict], 
                                    initial_capital: float) -> Dict:
        """포괄적인 성과 지표 계산
        
        Args:
            equity_curve: 자산 곡선 데이터
            trades: 거래 리스트
            initial_capital: 초기 자본금
        
        Returns:
            성과 지표 딕셔너리
        """
        # DataFrame 변환
        equity_df = pd.DataFrame(equity_curve)
        equity_df['time'] = pd.to_datetime(equity_df['time'])
        equity_df.set_index('time', inplace=True)
        
        # 기본 지표
        metrics = self._calculate_basic_metrics(equity_df, trades, initial_capital)
        
        # 위험 조정 지표
        risk_metrics = self._calculate_risk_adjusted_metrics(equity_df, initial_capital)
        metrics.update(risk_metrics)
        
        # 거래 통계
        trade_stats = self._calculate_trade_statistics(trades)
        metrics.update(trade_stats)
        
        # 시간대별 분석
        time_analysis = self._analyze_time_performance(trades)
        metrics['time_analysis'] = time_analysis
        
        # 연속 승/패 분석
        streak_analysis = self._analyze_streaks(trades)
        metrics['streak_analysis'] = streak_analysis
        
        # 월별 성과
        monthly_performance = self._calculate_monthly_performance(equity_df, initial_capital)
        metrics['monthly_performance'] = monthly_performance
        
        return metrics
    
    def _calculate_basic_metrics(self, equity_df: pd.DataFrame, trades: List[Dict], 
                               initial_capital: float) -> Dict:
        """기본 성과 지표 계산"""
        final_equity = equity_df['equity'].iloc[-1]
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # 거래 기본 통계
        total_trades = len(trades)
        if total_trades == 0:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_days_in_trade': 0
            }
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / total_trades
        
        # 평균 거래 기간
        durations = []
        for trade in trades:
            if trade.get('duration'):
                # duration이 문자열 형태일 수 있음
                if isinstance(trade['duration'], str):
                    # "1 days 02:30:00" 형태를 파싱
                    parts = trade['duration'].split(' days ')
                    if len(parts) == 2:
                        days = int(parts[0])
                        time_parts = parts[1].split(':')
                        hours = int(time_parts[0])
                        total_hours = days * 24 + hours
                        durations.append(total_hours / 24)  # 일 단위로 변환
                    else:
                        durations.append(1)  # 기본값
        
        avg_days_in_trade = np.mean(durations) if durations else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_days_in_trade': avg_days_in_trade
        }
    
    def _calculate_risk_adjusted_metrics(self, equity_df: pd.DataFrame, 
                                       initial_capital: float) -> Dict:
        """위험 조정 성과 지표 계산"""
        # 일간 수익률
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
        
        # 샤프 비율 (연율화)
        returns_mean = equity_df['returns'].mean()
        returns_std = equity_df['returns'].std()
        sharpe_ratio = np.sqrt(252) * returns_mean / returns_std if returns_std > 0 else 0
        
        # 소르티노 비율 (하방 위험만 고려)
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_std = downside_returns.std()
        sortino_ratio = np.sqrt(252) * returns_mean / downside_std if downside_std > 0 else 0
        
        # 칼마 비율 (연수익률 / 최대낙폭)
        annual_return = (equity_df['equity'].iloc[-1] / initial_capital) ** (252 / len(equity_df)) - 1
        
        # 최대 낙폭
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = abs(equity_df['drawdown'].min())
        
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 최대 낙폭 기간
        drawdown_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i in range(len(equity_df)):
            if equity_df['drawdown'].iloc[i] < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_dd_duration = i - drawdown_start
            else:
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
                drawdown_start = None
                current_dd_duration = 0
        
        # Value at Risk (95% 신뢰수준)
        var_95 = np.percentile(equity_df['returns'], 5)
        
        # Expected Shortfall (CVaR)
        cvar_95 = equity_df['returns'][equity_df['returns'] <= var_95].mean()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,
            'max_drawdown_duration_days': max_dd_duration,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': returns_std * np.sqrt(252) * 100
        }
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """거래 통계 계산"""
        if not trades:
            return {
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_rr_ratio': 0
            }
        
        profits = [t['profit'] for t in trades]
        winning_profits = [p for p in profits if p > 0]
        losing_profits = [p for p in profits if p < 0]
        
        # 평균 수익/손실
        avg_win = np.mean(winning_profits) if winning_profits else 0
        avg_loss = np.mean(losing_profits) if losing_profits else 0
        
        # 최대 수익/손실
        largest_win = max(winning_profits) if winning_profits else 0
        largest_loss = min(losing_profits) if losing_profits else 0
        
        # Profit Factor
        gross_profit = sum(winning_profits) if winning_profits else 0
        gross_loss = abs(sum(losing_profits)) if losing_profits else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 기대값 (Expectancy)
        win_rate = len(winning_profits) / len(trades) if trades else 0
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        # 평균 리스크/리워드 비율
        avg_rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 수익률 분포
        profit_percentages = [t['profit_pct'] for t in trades if 'profit_pct' in t]
        
        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_rr_ratio': avg_rr_ratio,
            'profit_std': np.std(profits) if profits else 0,
            'profit_skew': pd.Series(profits).skew() if len(profits) > 2 else 0,
            'profit_kurtosis': pd.Series(profits).kurtosis() if len(profits) > 3 else 0,
            'median_profit': np.median(profits) if profits else 0
        }
    
    def _analyze_time_performance(self, trades: List[Dict]) -> Dict:
        """시간대별 성과 분석"""
        if not trades:
            return {}
        
        # 거래를 시간대별로 분류
        hourly_performance = {}
        daily_performance = {}
        
        for trade in trades:
            if 'entry_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                hour = entry_time.hour
                day_of_week = entry_time.dayofweek
                
                # 시간별
                if hour not in hourly_performance:
                    hourly_performance[hour] = {'trades': 0, 'profit': 0, 'wins': 0}
                
                hourly_performance[hour]['trades'] += 1
                hourly_performance[hour]['profit'] += trade['profit']
                if trade['profit'] > 0:
                    hourly_performance[hour]['wins'] += 1
                
                # 요일별
                if day_of_week not in daily_performance:
                    daily_performance[day_of_week] = {'trades': 0, 'profit': 0, 'wins': 0}
                
                daily_performance[day_of_week]['trades'] += 1
                daily_performance[day_of_week]['profit'] += trade['profit']
                if trade['profit'] > 0:
                    daily_performance[day_of_week]['wins'] += 1
        
        # 승률 계산
        for hour, stats in hourly_performance.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        
        for day, stats in daily_performance.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        
        return {
            'hourly_performance': hourly_performance,
            'daily_performance': daily_performance
        }
    
    def _analyze_streaks(self, trades: List[Dict]) -> Dict:
        """연속 승/패 분석"""
        if not trades:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0
            }
        
        # 연속 승/패 계산
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        
        for trade in trades:
            if trade['profit'] > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'current_streak': current_streak,
            'avg_wins_per_streak': max_win_streak / max(max_loss_streak, 1)
        }
    
    def _calculate_monthly_performance(self, equity_df: pd.DataFrame, 
                                     initial_capital: float) -> List[Dict]:
        """월별 성과 계산"""
        monthly_data = []
        
        # 월별 그룹화
        monthly_equity = equity_df.resample('M').last()
        
        prev_equity = initial_capital
        for date, row in monthly_equity.iterrows():
            monthly_return = (row['equity'] - prev_equity) / prev_equity * 100
            monthly_data.append({
                'month': date.strftime('%Y-%m'),
                'equity': row['equity'],
                'return_pct': monthly_return
            })
            prev_equity = row['equity']
        
        return monthly_data
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """성과 보고서 생성"""
        report = []
        report.append("=" * 60)
        report.append("📊 전략 성과 보고서")
        report.append("=" * 60)
        
        # 기본 성과
        report.append("\n[기본 성과 지표]")
        report.append(f"총 수익률: {metrics['total_return_pct']:.2f}%")
        report.append(f"총 거래 수: {metrics['total_trades']}")
        report.append(f"승률: {metrics['win_rate']*100:.1f}%")
        report.append(f"평균 거래 기간: {metrics['avg_days_in_trade']:.1f}일")
        
        # 위험 조정 지표
        report.append("\n[위험 조정 성과]")
        report.append(f"샤프 비율: {metrics['sharpe_ratio']:.2f}")
        report.append(f"소르티노 비율: {metrics['sortino_ratio']:.2f}")
        report.append(f"칼마 비율: {metrics['calmar_ratio']:.2f}")
        report.append(f"최대 낙폭: {metrics['max_drawdown']:.1f}%")
        report.append(f"연간 변동성: {metrics['annual_volatility']:.1f}%")
        
        # 거래 통계
        report.append("\n[거래 통계]")
        report.append(f"평균 수익: ${metrics['avg_win']:.2f}")
        report.append(f"평균 손실: ${metrics['avg_loss']:.2f}")
        report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"기대값: ${metrics['expectancy']:.2f}")
        report.append(f"리스크/리워드 비율: {metrics['avg_rr_ratio']:.2f}")
        
        # 연속 승/패
        if 'streak_analysis' in metrics:
            report.append("\n[연속 승/패 분석]")
            report.append(f"최대 연속 승리: {metrics['streak_analysis']['max_consecutive_wins']}")
            report.append(f"최대 연속 패배: {metrics['streak_analysis']['max_consecutive_losses']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

# 테스트 함수
def test_performance_calculator():
    """성과 계산기 테스트"""
    print("=" * 60)
    print("성과 계산기 테스트")
    print("=" * 60)
    
    # 샘플 데이터 생성
    # 자산 곡선
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    initial_capital = 10000
    returns = np.random.normal(0.001, 0.02, 100)  # 일일 수익률
    
    equity_curve = []
    equity = initial_capital
    for i, date in enumerate(dates):
        equity *= (1 + returns[i])
        equity_curve.append({
            'time': date,
            'equity': equity,
            'capital': equity
        })
    
    # 샘플 거래
    trades = []
    for i in range(20):
        entry_price = 40000 + np.random.uniform(-1000, 1000)
        exit_price = entry_price + np.random.uniform(-500, 500)
        profit = exit_price - entry_price
        
        trades.append({
            'entry_time': dates[i*5].strftime('%Y-%m-%d %H:%M'),
            'exit_time': dates[i*5 + 3].strftime('%Y-%m-%d %H:%M'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'profit_pct': (profit / entry_price) * 100,
            'duration': '3 days 00:00:00'
        })
    
    # 성과 계산
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_performance_metrics(equity_curve, trades, initial_capital)
    
    # 보고서 생성
    report = analyzer.generate_performance_report(metrics)
    print(report)
    
    # 월별 성과
    if 'monthly_performance' in metrics:
        print("\n[월별 성과]")
        for month_data in metrics['monthly_performance'][:5]:  # 처음 5개월만
            print(f"{month_data['month']}: {month_data['return_pct']:.1f}%")

if __name__ == "__main__":
    test_performance_calculator()
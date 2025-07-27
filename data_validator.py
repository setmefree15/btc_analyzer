"""
데이터 검증 및 정리 모듈
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import utils

logger = utils.setup_logger(__name__)

class DataValidator:
    """데이터 검증 및 정리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.validation_results = {}
    
    def validate_data(self, df: pd.DataFrame, timeframe: str = '') -> Tuple[bool, Dict]:
        """데이터 검증"""
        results = {
            'timeframe': timeframe,
            'total_rows': len(df),
            'issues': [],
            'warnings': [],
            'is_valid': True
        }
        
        if df.empty:
            results['issues'].append("데이터프레임이 비어있습니다.")
            results['is_valid'] = False
            return False, results
        
        # 1. 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['issues'].append(f"필수 컬럼 누락: {missing_columns}")
            results['is_valid'] = False
            return False, results
        
        # 2. 인덱스 검증 (timestamp)
        if not isinstance(df.index, pd.DatetimeIndex):
            results['issues'].append("인덱스가 DatetimeIndex가 아닙니다.")
            results['is_valid'] = False
        
        # 3. NULL 값 확인
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            results['warnings'].append(f"NULL 값 발견: {null_counts[null_counts > 0].to_dict()}")
        
        # 4. 가격 관계 검증
        price_issues = []
        
        # high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            price_issues.append(f"high < low: {invalid_hl.sum()}개")
        
        # high >= open, close
        invalid_ho = df['high'] < df['open']
        invalid_hc = df['high'] < df['close']
        if invalid_ho.any() or invalid_hc.any():
            price_issues.append(f"high < open/close: {invalid_ho.sum() + invalid_hc.sum()}개")
        
        # low <= open, close
        invalid_lo = df['low'] > df['open']
        invalid_lc = df['low'] > df['close']
        if invalid_lo.any() or invalid_lc.any():
            price_issues.append(f"low > open/close: {invalid_lo.sum() + invalid_lc.sum()}개")
        
        if price_issues:
            results['issues'].extend(price_issues)
            results['is_valid'] = False
        
        # 5. 음수 값 확인
        negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any().any()
        negative_volume = (df['volume'] < 0).any()
        
        if negative_prices:
            results['issues'].append("음수 가격 발견")
            results['is_valid'] = False
        
        if negative_volume:
            results['warnings'].append("음수 거래량 발견")
        
        # 6. 중복 인덱스 확인
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"중복 인덱스: {duplicates}개")
        
        # 7. 시간 간격 확인
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            expected_freq = self._get_expected_frequency(timeframe)
            
            if expected_freq:
                irregular = (time_diffs.dropna() != expected_freq).sum()
                if irregular > 0:
                    results['warnings'].append(f"불규칙한 시간 간격: {irregular}개")
        
        # 8. 극단값 확인
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            if outliers > 0:
                results['warnings'].append(f"{col} 극단값: {outliers}개")
        
        # 결과 저장
        self.validation_results[timeframe] = results
        
        return results['is_valid'], results
    
    def clean_data(self, df: pd.DataFrame, fix_issues: bool = True) -> pd.DataFrame:
        """데이터 정리"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        logger.info("데이터 정리 시작...")
        
        # 1. 중복 인덱스 제거
        if df_clean.index.duplicated().any():
            before = len(df_clean)
            df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
            logger.info(f"중복 제거: {before - len(df_clean)}개")
        
        # 2. 정렬
        df_clean.sort_index(inplace=True)
        
        # 3. NULL 값 처리
        if df_clean.isnull().any().any():
            # Forward fill 후 backward fill
            df_clean.fillna(method='ffill', inplace=True)
            df_clean.fillna(method='bfill', inplace=True)
            
            # 여전히 NULL이 있다면 제거
            if df_clean.isnull().any().any():
                before = len(df_clean)
                df_clean.dropna(inplace=True)
                logger.info(f"NULL 행 제거: {before - len(df_clean)}개")
        
        if fix_issues:
            # 4. 가격 관계 수정
            # high는 open, close, low의 최댓값 이상이어야 함
            df_clean['high'] = df_clean[['high', 'open', 'close']].max(axis=1)
            
            # low는 open, close, high의 최솟값 이하여야 함
            df_clean['low'] = df_clean[['low', 'open', 'close']].min(axis=1)
            
            # 5. 음수 값 처리
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df_clean[col] = df_clean[col].abs()
            
            df_clean['volume'] = df_clean['volume'].abs()
        
        logger.info(f"데이터 정리 완료: {len(df_clean)} 행")
        
        return df_clean
    
    def _get_expected_frequency(self, timeframe: str) -> pd.Timedelta:
        """타임프레임별 예상 주기"""
        freq_map = {
            '15m': pd.Timedelta(minutes=15),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }
        return freq_map.get(timeframe)
    
    def generate_report(self) -> str:
        """검증 보고서 생성"""
        report = []
        report.append("=" * 60)
        report.append("데이터 검증 보고서")
        report.append("=" * 60)
        
        for tf, results in self.validation_results.items():
            report.append(f"\n타임프레임: {tf}")
            report.append(f"총 행 수: {results['total_rows']:,}")
            report.append(f"유효성: {'✅ 통과' if results['is_valid'] else '❌ 실패'}")
            
            if results['issues']:
                report.append("\n⚠️ 오류:")
                for issue in results['issues']:
                    report.append(f"  - {issue}")
            
            if results['warnings']:
                report.append("\n⚡ 경고:")
                for warning in results['warnings']:
                    report.append(f"  - {warning}")
            
            report.append("-" * 40)
        
        return "\n".join(report)
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """데이터 통계 정보"""
        if df.empty:
            return {}
        
        stats = {
            'period': {
                'start': df.index.min().strftime('%Y-%m-%d %H:%M'),
                'end': df.index.max().strftime('%Y-%m-%d %H:%M'),
                'days': (df.index.max() - df.index.min()).days
            },
            'rows': len(df),
            'price_stats': {},
            'volume_stats': {
                'total': df['volume'].sum(),
                'daily_avg': df['volume'].resample('D').sum().mean(),
                'max': df['volume'].max(),
                'min': df['volume'].min()
            }
        }
        
        # 가격 통계
        for col in ['open', 'high', 'low', 'close']:
            stats['price_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'current': df[col].iloc[-1]
            }
        
        # 변동성
        stats['volatility'] = {
            'daily_returns_std': df['close'].pct_change().std(),
            'annualized_vol': df['close'].pct_change().std() * np.sqrt(365)
        }
        
        return stats

# 테스트 함수
def test_validator():
    """검증기 테스트"""
    # 테스트 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    test_data = pd.DataFrame({
        'open': np.random.uniform(40000, 42000, 100),
        'high': np.random.uniform(41000, 43000, 100),
        'low': np.random.uniform(39000, 41000, 100),
        'close': np.random.uniform(40000, 42000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }, index=dates)
    
    # 일부러 문제 있는 데이터 추가
    test_data.loc[test_data.index[10], 'high'] = 35000  # high < low
    test_data.loc[test_data.index[20], 'volume'] = -100  # 음수 거래량
    test_data.loc[test_data.index[30], 'close'] = np.nan  # NULL 값
    
    validator = DataValidator()
    
    print("검증 전 데이터:")
    print(test_data.iloc[[10, 20, 30]])
    
    # 검증
    is_valid, results = validator.validate_data(test_data, '1h')
    print(f"\n검증 결과: {'통과' if is_valid else '실패'}")
    print(f"오류: {results['issues']}")
    print(f"경고: {results['warnings']}")
    
    # 정리
    cleaned_data = validator.clean_data(test_data, fix_issues=True)
    
    print("\n정리 후 데이터:")
    print(cleaned_data.iloc[[10, 20, 30]])
    
    # 통계
    stats = validator.get_data_statistics(cleaned_data)
    print(f"\n데이터 통계:")
    print(f"기간: {stats['period']['start']} ~ {stats['period']['end']}")
    print(f"현재 가격: ${stats['price_stats']['close']['current']:,.2f}")

if __name__ == "__main__":
    test_validator()
"""
데이터 수집 테스트 스크립트
"""
from data_collector import BTCDataCollector
from data_validator import DataValidator
import pandas as pd

def test_quick_collection():
    """빠른 데이터 수집 테스트 (최근 7일)"""
    print("=" * 60)
    print("BTC 데이터 수집 테스트")
    print("=" * 60)
    
    collector = BTCDataCollector()
    validator = DataValidator()
    
    # 최근 7일 데이터 수집
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=7)
    
    print(f"\n수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    # 1시간봉 데이터만 테스트
    timeframe = '1h'
    print(f"\n테스트 타임프레임: {timeframe}")
    
    try:
        # 데이터 수집
        print("데이터 수집 중...")
        df = collector.fetch_btc_data(
            timeframe,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not df.empty:
            print(f"✅ 수집 성공: {len(df)} 개의 캔들")
            
            # 데이터 검증
            print("\n데이터 검증 중...")
            is_valid, results = validator.validate_data(df, timeframe)
            
            if is_valid:
                print("✅ 데이터 검증 통과")
            else:
                print("❌ 데이터 검증 실패")
                print(f"오류: {results['issues']}")
                print(f"경고: {results['warnings']}")
                
                # 데이터 정리
                print("\n데이터 정리 중...")
                df = validator.clean_data(df, fix_issues=True)
                print("✅ 데이터 정리 완료")
            
            # 통계 표시
            stats = validator.get_data_statistics(df)
            print("\n📊 데이터 통계:")
            print(f"기간: {stats['period']['start']} ~ {stats['period']['end']}")
            print(f"현재가: ${stats['price_stats']['close']['current']:,.2f}")
            print(f"최고가: ${stats['price_stats']['high']['max']:,.2f}")
            print(f"최저가: ${stats['price_stats']['low']['min']:,.2f}")
            print(f"평균 거래량: {stats['volume_stats']['daily_avg']:,.2f}")
            
            # CSV 저장
            print("\nCSV 파일로 저장 중...")
            filepath = collector.save_to_csv(df, timeframe)
            if filepath:
                print(f"✅ 저장 완료: {filepath}")
            else:
                print("❌ 저장 실패")
                
        else:
            print("❌ 데이터 수집 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("\n가능한 원인:")
        print("1. 인터넷 연결 확인")
        print("2. 바이낸스 API 상태 확인")
        print("3. ccxt 라이브러리 설치 확인: pip install ccxt")

def test_load_data():
    """저장된 데이터 로드 테스트"""
    print("\n" + "=" * 60)
    print("저장된 데이터 로드 테스트")
    print("=" * 60)
    
    collector = BTCDataCollector()
    
    for tf in ['15m', '1h', '4h', '1d']:
        df = collector.load_data(tf)
        if not df.empty:
            print(f"\n{tf}: ✅ {len(df):,} 개의 캔들")
            print(f"  기간: {df.index.min()} ~ {df.index.max()}")
            print(f"  현재가: ${df['close'].iloc[-1]:,.2f}")
        else:
            print(f"\n{tf}: ❌ 데이터 없음")

if __name__ == "__main__":
    # 1. 빠른 수집 테스트
    test_quick_collection()
    
    # 2. 데이터 로드 테스트
    test_load_data()
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("스트림릿에서 데이터 수집 페이지를 사용해보세요.")
    print("=" * 60)
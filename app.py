"""
BTC 전용 분석 시스템 - 메인 앱
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import json

# 로컬 모듈
import config
import utils
from data_collector import BTCDataCollector
from data_validator import DataValidator
from indicators import TechnicalIndicators
from indicator_combinations import IndicatorCombinations
from backtester import StrategyBacktester
from strategy_optimizer import StrategyOptimizer
from performance_calculator import PerformanceAnalyzer

# 페이지 설정
st.set_page_config(
    page_title=config.STREAMLIT_CONFIG['page_title'],
    page_icon=config.STREAMLIT_CONFIG['page_icon'],
    layout=config.STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=config.STREAMLIT_CONFIG['initial_sidebar_state']
)

# 로거 설정
logger = utils.setup_logger(__name__, config.LOG_LEVEL)

# 세션 상태 초기화
def init_session_state():
    """세션 상태 초기화"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

# 사이드바
def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.title("🚀 BTC 분석 시스템")
        st.divider()
        
        # 네비게이션
        st.subheader("📍 네비게이션")
        pages = {
            'main': '🏠 메인 대시보드',
            'data_collection': '📊 데이터 수집',
            'indicator_analysis': '📈 지표 분석',
            'fractal_analysis': '🔄 프랙탈 분석',
            'chart_viewer': '📉 차트 뷰어'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.divider()
        
        # 시스템 상태
        st.subheader("⚙️ 시스템 상태")
        
        # 데이터 상태
        data_status = "✅ 로드됨" if st.session_state.data_loaded else "❌ 미로드"
        st.info(f"데이터: {data_status}")
        
        # 분석 상태
        analysis_count = len(st.session_state.analysis_results)
        st.info(f"분석 결과: {analysis_count}개")
        
        # 현재 시간
        st.info(f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.divider()
        
        # 정보
        st.caption("BTC 전용 기술적 분석 시스템")
        st.caption("v1.0.0")

# 메인 대시보드
def render_main_dashboard():
    """메인 대시보드 렌더링"""
    st.title("🏠 메인 대시보드")
    st.divider()
    
    # 시스템 소개
    st.markdown("""
    ### 🎯 BTC 전용 분석 시스템
    
    이 시스템은 비트코인(BTC) 전용 기술적 분석 도구입니다.
    
    **주요 기능:**
    - 📊 **데이터 수집**: 바이낸스에서 BTC/USDT 데이터 수집
    - 📈 **지표 분석**: RSI, MACD, 볼린저밴드 등 기술적 지표 최적화
    - 🔄 **프랙탈 분석**: DTW 알고리즘을 이용한 패턴 매칭
    - 📉 **차트 뷰어**: 인터랙티브 차트 및 지표 시각화
    """)
    
    st.divider()
    
    # 빠른 시작
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 1️⃣ 데이터 수집")
        st.write("먼저 BTC 데이터를 수집하세요")
        if st.button("데이터 수집 시작", key="quick_data"):
            st.session_state.current_page = 'data_collection'
            st.rerun()
    
    with col2:
        st.success("### 2️⃣ 지표 분석")
        st.write("기술적 지표를 분석하세요")
        if st.button("지표 분석 시작", key="quick_indicator"):
            st.session_state.current_page = 'indicator_analysis'
            st.rerun()
    
    with col3:
        st.warning("### 3️⃣ 프랙탈 분석")
        st.write("패턴 매칭을 수행하세요")
        if st.button("프랙탈 분석 시작", key="quick_fractal"):
            st.session_state.current_page = 'fractal_analysis'
            st.rerun()
    
    st.divider()
    
    # 현재 상태 요약
    st.subheader("📊 현재 상태")
    
    # 데이터 파일 확인
    collector = BTCDataCollector()
    data_files = []
    latest_prices = {}
    
    for tf in config.TIMEFRAMES.keys():
        filepath = os.path.join(config.DATA_DIR, f"btc_usdt_{tf}.csv")
        if os.path.exists(filepath):
            data_files.append(tf)
            # 최신 가격 정보 가져오기
            try:
                df = collector.load_data(tf)
                if not df.empty:
                    latest_prices[tf] = {
                        'price': df['close'].iloc[-1],
                        'time': df.index[-1],
                        'change': ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) > 1 else 0
                    }
            except:
                pass
    
    if data_files:
        st.success(f"✅ 사용 가능한 데이터: {', '.join([config.TIMEFRAMES[tf] for tf in data_files])}")
        
        # 현재 가격 표시
        if latest_prices:
            st.divider()
            st.subheader("💰 현재 BTC 가격")
            
            cols = st.columns(len(latest_prices))
            for idx, (tf, price_info) in enumerate(latest_prices.items()):
                with cols[idx]:
                    st.metric(
                        label=config.TIMEFRAMES[tf],
                        value=f"${price_info['price']:,.2f}",
                        delta=f"{price_info['change']:.2f}%",
                        help=f"마지막 업데이트: {price_info['time'].strftime('%Y-%m-%d %H:%M')}"
                    )
    else:
        st.error("❌ 수집된 데이터가 없습니다. 데이터 수집을 먼저 진행하세요.")
    
    # 분석 결과 확인
    result_files = os.listdir(config.RESULTS_DIR) if os.path.exists(config.RESULTS_DIR) else []
    if result_files:
        st.success(f"✅ 분석 결과 파일: {len(result_files)}개")
    else:
        st.warning("⚠️ 아직 분석 결과가 없습니다.")

# 데이터 수집 페이지
def render_data_collection():
    """데이터 수집 페이지 렌더링"""
    st.title("📊 데이터 수집")
    st.divider()
    
    st.info("""
    **바이낸스 거래소에서 BTC/USDT 데이터를 수집합니다.**
    - 기간: 2021년 1월 ~ 현재
    - 타임프레임: 15분, 1시간, 4시간, 일봉
    """)
    
    # 수집 설정
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=pd.to_datetime(config.START_DATE),
            min_value=pd.to_datetime('2020-01-01'),
            max_value=pd.to_datetime('today')
        )
    
    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=pd.to_datetime('today'),
            min_value=start_date,
            max_value=pd.to_datetime('today')
        )
    
    # 타임프레임 선택
    selected_timeframes = st.multiselect(
        "수집할 타임프레임",
        options=list(config.TIMEFRAMES.keys()),
        default=list(config.TIMEFRAMES.keys()),
        format_func=lambda x: config.TIMEFRAMES[x]
    )
    
    st.divider()
    
    # 수집 시작 버튼
    if st.button("🚀 데이터 수집 시작", type="primary", use_container_width=True):
        if not selected_timeframes:
            st.error("타임프레임을 선택해주세요.")
            return
        
        # 데이터 수집기 초기화
        collector = BTCDataCollector()
        validator = DataValidator()
        
        # 진행 상태 컨테이너
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        # 각 타임프레임별로 수집
        total_timeframes = len(selected_timeframes)
        collected_data = {}
        
        for idx, tf in enumerate(selected_timeframes):
            progress = (idx / total_timeframes)
            progress_bar.progress(progress)
            progress_text.text(f"수집 중... {config.TIMEFRAMES[tf]} ({idx + 1}/{total_timeframes})")
            
            with status_container:
                with st.spinner(f"{config.TIMEFRAMES[tf]} 데이터 수집 중..."):
                    try:
                        # 데이터 수집
                        df = collector.fetch_btc_data(
                            tf, 
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if not df.empty:
                            # 데이터 검증
                            is_valid, validation_results = validator.validate_data(df, tf)
                            
                            if not is_valid:
                                st.warning(f"{config.TIMEFRAMES[tf]}: 데이터 검증 실패 - 자동 정리 중...")
                                df = validator.clean_data(df, fix_issues=True)
                            
                            # 데이터 저장
                            filepath = collector.save_to_csv(df, tf)
                            
                            if filepath:
                                collected_data[tf] = df
                                st.success(f"✅ {config.TIMEFRAMES[tf]}: {len(df):,}개 캔들 수집 완료")
                            else:
                                st.error(f"❌ {config.TIMEFRAMES[tf]}: 저장 실패")
                        else:
                            st.error(f"❌ {config.TIMEFRAMES[tf]}: 데이터 수집 실패")
                            
                    except Exception as e:
                        st.error(f"❌ {config.TIMEFRAMES[tf]}: 오류 발생 - {str(e)}")
        
        # 완료
        progress_bar.progress(1.0)
        progress_text.text("수집 완료!")
        
        # 결과 요약
        if collected_data:
            st.divider()
            st.subheader("📊 수집 결과")
            
            summary_data = []
            for tf, df in collected_data.items():
                stats = validator.get_data_statistics(df)
                summary_data.append({
                    '타임프레임': config.TIMEFRAMES[tf],
                    '캔들 수': f"{len(df):,}",
                    '시작': stats['period']['start'],
                    '종료': stats['period']['end'],
                    '현재가': f"${stats['price_stats']['close']['current']:,.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # 세션 상태 업데이트
            st.session_state.data_loaded = True
            st.balloons()
    
    # 빠른 수집 버튼들
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ 최근 7일 수집", use_container_width=True):
            # 7일 전부터 오늘까지 자동 설정
            st.session_state['quick_collect'] = '7d'
            st.rerun()
    
    with col2:
        if st.button("⚡ 최근 30일 수집", use_container_width=True):
            st.session_state['quick_collect'] = '30d'
            st.rerun()
    
    with col3:
        if st.button("🔄 데이터 업데이트", use_container_width=True):
            st.session_state['update_data'] = True
            st.rerun()
    
    # 빠른 수집 처리
    if 'quick_collect' in st.session_state:
        days = 7 if st.session_state['quick_collect'] == '7d' else 30
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        collector = BTCDataCollector()
        
        with st.spinner(f"최근 {days}일 데이터 수집 중..."):
            for tf in config.TIMEFRAMES.keys():
                try:
                    df = collector.fetch_btc_data(
                        tf,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    if not df.empty:
                        collector.save_to_csv(df, tf)
                        st.success(f"✅ {config.TIMEFRAMES[tf]}: {len(df):,}개 캔들")
                except Exception as e:
                    st.error(f"❌ {config.TIMEFRAMES[tf]}: {str(e)}")
        
        del st.session_state['quick_collect']
        st.balloons()
    
    # 데이터 업데이트 처리
    if 'update_data' in st.session_state:
        collector = BTCDataCollector()
        
        with st.spinner("기존 데이터 업데이트 중..."):
            for tf in config.TIMEFRAMES.keys():
                try:
                    df = collector.update_data(tf)
                    if not df.empty:
                        st.success(f"✅ {config.TIMEFRAMES[tf]}: 업데이트 완료")
                except Exception as e:
                    st.error(f"❌ {config.TIMEFRAMES[tf]}: {str(e)}")
        
        del st.session_state['update_data']
        st.balloons()
    
    # 기존 데이터 표시
    st.divider()
    st.subheader("📁 기존 데이터")
    
    data_info = []
    validator = DataValidator()
    
    for tf in config.TIMEFRAMES.keys():
        filepath = os.path.join(config.DATA_DIR, f"btc_usdt_{tf}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
                stats = validator.get_data_statistics(df)
                
                data_info.append({
                    '타임프레임': config.TIMEFRAMES[tf],
                    '캔들 수': f"{len(df):,}",
                    '시작': stats['period']['start'],
                    '종료': stats['period']['end'],
                    '크기': f"{os.path.getsize(filepath) / 1024 / 1024:.2f} MB",
                    '상태': '✅'
                })
            except:
                data_info.append({
                    '타임프레임': config.TIMEFRAMES[tf],
                    '캔들 수': '-',
                    '시작': '-',
                    '종료': '-',
                    '크기': f"{os.path.getsize(filepath) / 1024 / 1024:.2f} MB",
                    '상태': '⚠️'
                })
        else:
            data_info.append({
                '타임프레임': config.TIMEFRAMES[tf],
                '캔들 수': '-',
                '시작': '-',
                '종료': '-',
                '크기': '-',
                '상태': '❌'
            })
    
    df_info = pd.DataFrame(data_info)
    st.dataframe(df_info, use_container_width=True, hide_index=True)

# 지표 분석 페이지
def render_indicator_analysis():
    """지표 분석 페이지 렌더링"""
    st.title("📈 지표 분석")
    st.divider()
    
    # 데이터 확인
    collector = BTCDataCollector()
    available_timeframes = []
    
    for tf in config.TIMEFRAMES.keys():
        if os.path.exists(os.path.join(config.DATA_DIR, f"btc_usdt_{tf}.csv")):
            available_timeframes.append(tf)
    
    if not available_timeframes:
        st.error("❌ 수집된 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        if st.button("데이터 수집 페이지로 이동"):
            st.session_state.current_page = 'data_collection'
            st.rerun()
        return
    
    # 분석 설정
    st.subheader("🎯 분석 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_timeframe = st.selectbox(
            "타임프레임",
            options=available_timeframes,
            format_func=lambda x: config.TIMEFRAMES[x]
        )
    
    with col2:
        analysis_period = st.selectbox(
            "분석 기간",
            options=['전체', '최근 1년', '최근 6개월', '최근 3개월', '최근 1개월'],
            index=4  # 기본값: 최근 1개월
        )
    
    # 지표 선택
    st.subheader("📊 분석할 지표 선택")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_rsi = st.checkbox("RSI", value=True)
        use_macd = st.checkbox("MACD", value=True)
    
    with col2:
        use_bb = st.checkbox("볼린저밴드", value=True)
        use_ma = st.checkbox("이동평균", value=True)
    
    with col3:
        use_stoch = st.checkbox("스토캐스틱", value=False)
        use_volume = st.checkbox("거래량 지표", value=False)
    
    # 분석 시작 버튼
    if st.button("🔍 지표 분석 시작", type="primary", use_container_width=True):
        with st.spinner("지표 계산 중..."):
            # 데이터 로드
            df = collector.load_data(selected_timeframe)
            
            if df.empty:
                st.error("데이터 로드 실패")
                return
            
            # 기간 필터링
            if analysis_period != '전체':
                end_date = df.index.max()
                if analysis_period == '최근 1년':
                    start_date = end_date - pd.Timedelta(days=365)
                elif analysis_period == '최근 6개월':
                    start_date = end_date - pd.Timedelta(days=180)
                elif analysis_period == '최근 3개월':
                    start_date = end_date - pd.Timedelta(days=90)
                else:  # 최근 1개월
                    start_date = end_date - pd.Timedelta(days=30)
                
                df = df[df.index >= start_date]
            
            # 지표 계산
            indicators = TechnicalIndicators()
            
            # 선택된 지표만 계산
            if use_rsi:
                df = indicators.calculate_rsi(df)
            if use_macd:
                df = indicators.calculate_macd(df)
            if use_bb:
                df = indicators.calculate_bollinger_bands(df)
            if use_ma:
                df = indicators.calculate_moving_averages(df)
            if use_stoch:
                df = indicators.calculate_stochastic(df)
            if use_volume:
                df = indicators.calculate_volume_indicators(df)
            
            # 거래 신호 추가
            df = indicators.add_trading_signals(df)
            
            # 결과를 세션에 저장
            st.session_state['indicator_results'] = {
                'data': df,
                'timeframe': selected_timeframe,
                'period': analysis_period,
                'timestamp': datetime.now()
            }
    
    # 분석 결과 표시
    if 'indicator_results' in st.session_state:
        results = st.session_state['indicator_results']
        df = results['data']
        
        st.divider()
        st.subheader("📊 분석 결과")
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["📈 차트", "📊 지표 값", "🚨 신호", "📋 요약"])
        
        with tab1:
            # 차트 표시
            st.subheader("가격 및 지표 차트")
            
            # 메인 차트 (가격 + 이동평균)
            fig = go.Figure()
            
            # 캔들스틱
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='BTC/USDT'
            ))
            
            # 이동평균
            if 'EMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['EMA_20'],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color='orange', width=2)
                ))
            
            if 'EMA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['EMA_50'],
                    mode='lines',
                    name='EMA 50',
                    line=dict(color='blue', width=2)
                ))
            
            # 볼린저 밴드
            if 'BB_upper_2_0' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_upper_2_0'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_lower_2_0'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ))
            
            fig.update_layout(
                title=f'BTC/USDT {config.TIMEFRAMES[results["timeframe"]]} - {results["period"]}',
                yaxis_title='가격 (USDT)',
                xaxis_title='시간',
                height=600,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI 차트
            if 'RSI_14' in df.columns:
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI_14'],
                    mode='lines',
                    name='RSI 14',
                    line=dict(color='purple', width=2)
                ))
                
                # 과매수/과매도 영역
                fig_rsi.add_hline(y=70, line_color="red", line_dash="dash", annotation_text="과매수")
                fig_rsi.add_hline(y=30, line_color="green", line_dash="dash", annotation_text="과매도")
                
                fig_rsi.update_layout(
                    title='RSI (Relative Strength Index)',
                    yaxis_title='RSI',
                    xaxis_title='시간',
                    height=300,
                    template='plotly_dark',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD 차트
            if 'MACD_12_26_9' in df.columns:
                fig_macd = go.Figure()
                
                fig_macd.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD_12_26_9'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD_signal_12_26_9'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red', width=2)
                ))
                
                fig_macd.add_trace(go.Bar(
                    x=df.index,
                    y=df['MACD_hist_12_26_9'],
                    name='Histogram',
                    marker_color='gray'
                ))
                
                fig_macd.update_layout(
                    title='MACD (Moving Average Convergence Divergence)',
                    yaxis_title='MACD',
                    xaxis_title='시간',
                    height=300,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab2:
            # 현재 지표 값
            st.subheader("현재 지표 값")
            
            latest_values = df.iloc[-1]
            
            # 주요 지표 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'RSI_14' in df.columns:
                    rsi_value = latest_values['RSI_14']
                    rsi_status = "과매수" if rsi_value > 70 else "과매도" if rsi_value < 30 else "중립"
                    st.metric("RSI 14", f"{rsi_value:.2f}", rsi_status)
            
            with col2:
                if 'MACD_12_26_9' in df.columns:
                    macd_value = latest_values['MACD_12_26_9']
                    signal_value = latest_values['MACD_signal_12_26_9']
                    macd_status = "상승" if macd_value > signal_value else "하락"
                    st.metric("MACD", f"{macd_value:.2f}", macd_status)
            
            with col3:
                if 'BB_percent_2_0' in df.columns:
                    bb_percent = latest_values['BB_percent_2_0'] * 100
                    bb_status = "상단" if bb_percent > 80 else "하단" if bb_percent < 20 else "중간"
                    st.metric("BB %", f"{bb_percent:.1f}%", bb_status)
            
            with col4:
                price = latest_values['close']
                st.metric("현재가", f"${price:,.2f}", "")
            
            # 상세 지표 테이블
            st.divider()
            st.subheader("상세 지표 값")
            
            # 지표 컬럼만 선택
            indicator_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # 최근 10개 데이터
            recent_data = df[['close'] + indicator_columns].tail(10)
            
            # 소수점 2자리로 포맷
            formatted_data = recent_data.round(2)
            
            st.dataframe(formatted_data, use_container_width=True)
        
        with tab3:
            # 거래 신호
            st.subheader("거래 신호")
            
            # 현재 활성 신호
            signal_columns = [col for col in df.columns if any(signal in col for signal in ['oversold', 'overbought', 'bullish', 'bearish', 'cross', 'breakout'])]
            
            active_signals = []
            for signal in signal_columns:
                if signal in df.columns and df[signal].iloc[-1] == 1:
                    active_signals.append(signal)
            
            if active_signals:
                st.success("🚨 현재 활성 신호:")
                for signal in active_signals:
                    st.write(f"- {signal}")
            else:
                st.info("현재 활성화된 신호가 없습니다.")
            
            # 최근 신호 이력
            st.divider()
            st.subheader("최근 신호 이력")
            
            signal_history = []
            for idx in range(max(0, len(df) - 50), len(df)):
                for signal in signal_columns:
                    if signal in df.columns and df[signal].iloc[idx] == 1:
                        signal_history.append({
                            '시간': df.index[idx],
                            '신호': signal,
                            '가격': df['close'].iloc[idx]
                        })
            
            if signal_history:
                signal_df = pd.DataFrame(signal_history)
                signal_df = signal_df.sort_values('시간', ascending=False).head(20)
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
            else:
                st.info("최근 신호가 없습니다.")
        
        with tab4:
            # 요약 통계
            st.subheader("분석 요약")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **분석 정보**
                - 타임프레임: {config.TIMEFRAMES[results['timeframe']]}
                - 분석 기간: {results['period']}
                - 데이터 개수: {len(df):,} 캔들
                - 분석 시간: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                """)
            
            with col2:
                # 가격 통계
                price_stats = {
                    '평균가': df['close'].mean(),
                    '최고가': df['high'].max(),
                    '최저가': df['low'].min(),
                    '변동성': df['close'].std()
                }
                
                st.success("**가격 통계**")
                for key, value in price_stats.items():
                    st.write(f"- {key}: ${value:,.2f}")
            
            # 지표별 현재 상태
            st.divider()
            st.subheader("지표별 현재 상태")
            
            indicator_summary = TechnicalIndicators().get_indicator_summary(df)
            
            # 지표 상태를 시각적으로 표시
            status_data = []
            
            if 'RSI_14' in df.columns:
                rsi_val = df['RSI_14'].iloc[-1]
                status_data.append({
                    '지표': 'RSI',
                    '값': f"{rsi_val:.2f}",
                    '상태': '🔴 과매수' if rsi_val > 70 else '🟢 과매도' if rsi_val < 30 else '🟡 중립'
                })
            
            if 'MACD_12_26_9' in df.columns:
                macd_diff = df['MACD_12_26_9'].iloc[-1] - df['MACD_signal_12_26_9'].iloc[-1]
                status_data.append({
                    '지표': 'MACD',
                    '값': f"{macd_diff:.2f}",
                    '상태': '🟢 상승' if macd_diff > 0 else '🔴 하락'
                })
            
            if status_data:
                status_df = pd.DataFrame(status_data)
                st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # 백테스팅 섹션
    st.divider()
    st.subheader("🎯 백테스팅 및 전략 최적화")
    
    with st.expander("백테스팅 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_capital = st.number_input(
                "초기 자본금 ($)",
                min_value=1000,
                max_value=1000000,
                value=config.BACKTEST_CONFIG['initial_capital'],
                step=1000
            )
        
        with col2:
            backtest_commission = st.number_input(
                "수수료 (%)",
                min_value=0.0,
                max_value=1.0,
                value=config.BACKTEST_CONFIG['commission'] * 100,
                step=0.01,
                format="%.3f"
            ) / 100
        
        with col3:
            backtest_slippage = st.number_input(
                "슬리피지 (%)",
                min_value=0.0,
                max_value=1.0,
                value=config.BACKTEST_CONFIG['slippage'] * 100,
                step=0.01,
                format="%.3f"
            ) / 100
        
        # 백테스팅 실행 버튼
        col1, col2 = st.columns(2)
        
        with col1:
            # 전략 선택
            strategy_type = st.selectbox(
                "테스트할 전략",
                ["RSI 단순", "MACD 크로스", "볼린저밴드", "이동평균 크로스"],
                help="다양한 전략을 테스트해보세요"
            )
        
        with col2:
            if st.button("🔍 단일 전략 백테스팅", type="secondary", use_container_width=True):
                if 'indicator_results' not in st.session_state:
                    st.error("먼저 지표 분석을 실행해주세요.")
                else:
                    with st.spinner("백테스팅 실행 중..."):
                        # 선택된 전략에 따라 다른 조건 설정
                        if strategy_type == "RSI 단순":
                            sample_strategy = {
                                'name': 'Simple_RSI_Strategy',
                                'type': 'momentum',
                                'entry_rules': ['RSI_14 < 45'],  # 더 완화된 조건
                                'exit_rules': ['RSI_14 > 55']    # 더 완화된 조건
                            }
                        elif strategy_type == "MACD 크로스":
                            sample_strategy = {
                                'name': 'MACD_Cross_Strategy',
                                'type': 'trend',
                                'entry_rules': ['MACD_12_26_9 > MACD_signal_12_26_9'],
                                'exit_rules': ['MACD_12_26_9 < MACD_signal_12_26_9']
                            }
                        elif strategy_type == "볼린저밴드":
                            sample_strategy = {
                                'name': 'BB_Strategy',
                                'type': 'volatility',
                                'entry_rules': ['close < BB_lower_2_0'],
                                'exit_rules': ['close > BB_middle_2_0']
                            }
                        else:  # 이동평균 크로스
                            sample_strategy = {
                                'name': 'MA_Cross_Strategy',
                                'type': 'trend',
                                'entry_rules': ['EMA_10 > EMA_20'],
                                'exit_rules': ['EMA_10 < EMA_20']
                            }
                        
                        # 백테스팅 실행
                        backtester = StrategyBacktester(
                            initial_capital=backtest_capital,
                            commission=backtest_commission,
                            slippage=backtest_slippage
                        )
                        
                        df = st.session_state['indicator_results']['data']
                        
                        # 지표 확인 (디버깅용)
                        required_columns = []
                        for rule in sample_strategy['entry_rules'] + sample_strategy['exit_rules']:
                            # 규칙에서 컬럼명 추출
                            for word in rule.split():
                                if word in df.columns:
                                    required_columns.append(word)
                        
                        missing_columns = []
                        for col in required_columns:
                            if col not in df.columns:
                                missing_columns.append(col)
                        
                        if missing_columns:
                            st.error(f"필요한 지표가 없습니다: {', '.join(missing_columns)}")
                            st.info("지표 분석을 다시 실행하고 필요한 지표를 선택해주세요.")
                            return
                        
                        # 지표 통계 표시
                        with st.expander("지표 통계 (디버깅)", expanded=False):
                            if 'RSI_14' in df.columns:
                                col1, col2, col3 = st.columns(3)
                                rsi_clean = df['RSI_14'].dropna()
                                
                                with col1:
                                    st.metric("RSI 최소값", f"{rsi_clean.min():.1f}")
                                    st.metric("45 미만 비율", f"{(rsi_clean < 45).sum() / len(rsi_clean) * 100:.1f}%")
                                
                                with col2:
                                    st.metric("RSI 평균", f"{rsi_clean.mean():.1f}")
                                    st.metric("55 초과 비율", f"{(rsi_clean > 55).sum() / len(rsi_clean) * 100:.1f}%")
                                
                                with col3:
                                    st.metric("RSI 최대값", f"{rsi_clean.max():.1f}")
                                    st.metric("유효 데이터", f"{len(rsi_clean)} / {len(df)}")
                        
                        results = backtester.backtest_strategy(df, sample_strategy)
                        
                        # 결과 저장
                        st.session_state['backtest_results'] = {
                            'single_strategy': results,
                            'equity_curve': backtester.equity_curve,
                            'timestamp': datetime.now()
                        }
        
        with col2:
            if st.button("🚀 전략 최적화 (다중 백테스팅)", type="primary", use_container_width=True):
                st.session_state['run_optimization'] = True
    
    # 단일 전략 백테스팅 결과
    if 'backtest_results' in st.session_state and 'single_strategy' in st.session_state['backtest_results']:
        st.divider()
        st.subheader("📊 백테스팅 결과")
        
        results = st.session_state['backtest_results']['single_strategy']
        
        # 거래가 없는 경우 경고 메시지
        if results['total_trades'] == 0:
            st.warning("""
            ⚠️ 거래가 발생하지 않았습니다.
            
            가능한 원인:
            - 진입/청산 조건이 너무 엄격함
            - 분석 기간이 너무 짧음
            - 지표가 조건을 만족하지 못함
            """)
            
            # 신호 발생 정보 표시
            if 'entry_signals' in results:
                st.info(f"진입 신호: {results['entry_signals']}회, 청산 신호: {results['exit_signals']}회")
                if results['entry_signals'] > 0 and results['exit_signals'] == 0:
                    st.info("💡 진입은 했지만 청산 조건을 만족하지 못했습니다. 청산 조건을 완화해보세요.")
                elif results['entry_signals'] == 0:
                    st.info("💡 진입 조건을 만족하는 경우가 없습니다. 진입 조건을 완화해보세요.")
        
        # 주요 지표 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 수익률",
                f"{results['total_return_pct']:.1f}%",
                f"${results['total_return']:,.2f}"
            )
        
        with col2:
            st.metric(
                "승률",
                f"{results['win_rate']*100:.1f}%",
                f"{results['winning_trades']}/{results['total_trades']}"
            )
        
        with col3:
            st.metric(
                "샤프 비율",
                f"{results['sharpe_ratio']:.2f}",
                "위험 조정 수익"
            )
        
        with col4:
            st.metric(
                "최대 낙폭",
                f"{results['max_drawdown']:.1f}%",
                "최대 손실"
            )
        
        # 자산 곡선 차트
        equity_curve = st.session_state['backtest_results']['equity_curve']
        equity_df = pd.DataFrame(equity_curve)
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='자산 가치',
            line=dict(color='green', width=2)
        ))
        
        fig_equity.add_hline(
            y=backtest_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="초기 자본"
        )
        
        fig_equity.update_layout(
            title='자산 곡선 (Equity Curve)',
            xaxis_title='시간',
            yaxis_title='자산 가치 ($)',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # 상세 통계
        with st.expander("상세 통계"):
            if results['total_trades'] == 0:
                st.warning("⚠️ 거래가 발생하지 않았습니다. 진입/청산 조건을 확인해주세요.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**거래 통계**")
                    st.write(f"- 총 거래 수: {results['total_trades']}")
                    st.write(f"- 평균 거래 수익: ${results['avg_trade_return']:.2f}")
                    st.write(f"- 평균 수익 거래: ${results['avg_win']:.2f}")
                    st.write(f"- 평균 손실 거래: ${results['avg_loss']:.2f}")
                    st.write(f"- Profit Factor: {results['profit_factor']:.2f}")
                
                with col2:
                    st.write("**성과 지표**")
                    if 'final_capital' in results:
                        st.write(f"- 최종 자본: ${results['final_capital']:,.2f}")
                    else:
                        st.write(f"- 최종 자본: ${backtest_capital:,.2f}")
                    st.write(f"- 샤프 비율: {results['sharpe_ratio']:.2f}")
                    st.write(f"- 최대 낙폭: {results['max_drawdown']:.1f}%")
                    
                    # 성과 분석기 사용
                    analyzer = PerformanceAnalyzer()
                    detailed_metrics = analyzer.calculate_performance_metrics(
                        equity_curve,
                        results['trades'],
                        backtest_capital
                    )
                    
                    if 'sortino_ratio' in detailed_metrics:
                        st.write(f"- 소르티노 비율: {detailed_metrics['sortino_ratio']:.2f}")
                    if 'calmar_ratio' in detailed_metrics:
                        st.write(f"- 칼마 비율: {detailed_metrics['calmar_ratio']:.2f}")
    
    # 전략 최적화 실행
    if 'run_optimization' in st.session_state and st.session_state['run_optimization']:
        st.divider()
        st.subheader("🚀 전략 최적화 진행")
        
        # 최적화 설정
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                max_indicators = st.slider(
                    "최대 지표 조합 수",
                    min_value=1,
                    max_value=3,
                    value=2
                )
            
            with col2:
                max_strategies = st.slider(
                    "테스트할 전략 수 (제한)",
                    min_value=10,
                    max_value=100,
                    value=20,
                    step=10
                )
            
            if st.button("최적화 시작", type="primary"):
                with st.spinner("전략 최적화 중... (시간이 걸릴 수 있습니다)"):
                    # 데이터 준비
                    df = st.session_state['indicator_results']['data']
                    
                    # 지표 조합 생성
                    combiner = IndicatorCombinations()
                    indicator_combos = combiner.generate_indicator_combinations(max_indicators)
                    
                    # 전략 생성 (제한된 수만큼)
                    all_strategies = []
                    for combo in indicator_combos[:max_strategies//3]:  # 조합당 약 3개 파라미터
                        param_combos = combiner.generate_parameter_combinations(combo)
                        for param_combo in param_combos[:3]:
                            strategy = combiner.create_strategy_config(combo, param_combo['parameters'])
                            all_strategies.append(strategy)
                            if len(all_strategies) >= max_strategies:
                                break
                        if len(all_strategies) >= max_strategies:
                            break
                    
                    # 최적화 실행
                    optimizer = StrategyOptimizer(initial_capital=backtest_capital)
                    optimization_results = optimizer.optimize_strategies(
                        df,
                        all_strategies,
                        selected_timeframe,
                        max_workers=1  # Windows 호환성을 위해 순차 처리
                    )
                    
                    # 결과 저장
                    st.session_state['optimization_results'] = optimization_results
                    
                    # 파일로 저장
                    filepath = optimizer.save_optimization_results(optimization_results)
                    st.success(f"✅ 최적화 완료! 결과 저장: {filepath}")
        
        # 전략 최적화 완료 후 플래그 리셋
        st.session_state['run_optimization'] = False
    
    # 최적화 결과 표시
    if 'optimization_results' in st.session_state:
        st.divider()
        st.subheader("🏆 최적화 결과")
        
        opt_results = st.session_state['optimization_results']
        
        # 요약 통계
        summary = opt_results['summary_stats']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "테스트 전략 수",
                f"{summary['total_strategies']}",
                "완료"
            )
        
        with col2:
            st.metric(
                "평균 수익률",
                f"{summary['avg_return']:.1f}%",
                ""
            )
        
        with col3:
            st.metric(
                "수익 전략 비율",
                f"{summary['profitable_ratio']*100:.0f}%",
                f"{summary['profitable_strategies']}개"
            )
        
        with col4:
            st.metric(
                "평균 샤프 비율",
                f"{summary['avg_sharpe']:.2f}",
                ""
            )
        
        # 최고 전략들
        st.subheader("🥇 최고 성과 전략")
        
        tabs = st.tabs(["종합 최고", "수익률 최고", "샤프 비율 최고", "승률 최고"])
        
        with tabs[0]:
            # 종합 최고 전략
            if opt_results['best_strategies']['overall_best']:
                for i, strategy in enumerate(opt_results['best_strategies']['overall_best'][:5]):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.write(f"**#{i+1} {strategy['name']}**")
                        
                        with col2:
                            st.write(f"수익률: {strategy['performance']['total_return_pct']:.1f}%")
                        
                        with col3:
                            st.write(f"샤프: {strategy['performance']['sharpe_ratio']:.2f}")
                        
                        with col4:
                            st.write(f"승률: {strategy['performance']['win_rate']*100:.0f}%")
                        
                        with col5:
                            st.write(f"거래: {strategy['performance']['total_trades']}")
        
        with tabs[1]:
            # 수익률 최고
            if 'best_by_total_return_pct' in opt_results['best_strategies']:
                for i, strategy in enumerate(opt_results['best_strategies']['best_by_total_return_pct'][:5]):
                    st.write(f"**#{i+1}** {strategy['name']}: {strategy['performance']['total_return_pct']:.1f}%")
        
        with tabs[2]:
            # 샤프 비율 최고
            if 'best_by_sharpe_ratio' in opt_results['best_strategies']:
                for i, strategy in enumerate(opt_results['best_strategies']['best_by_sharpe_ratio'][:5]):
                    st.write(f"**#{i+1}** {strategy['name']}: {strategy['performance']['sharpe_ratio']:.2f}")
        
        with tabs[3]:
            # 승률 최고
            if 'best_by_win_rate' in opt_results['best_strategies']:
                for i, strategy in enumerate(opt_results['best_strategies']['best_by_win_rate'][:5]):
                    st.write(f"**#{i+1}** {strategy['name']}: {strategy['performance']['win_rate']*100:.0f}%")

# 프랙탈 분석 페이지
def render_fractal_analysis():
    """프랙탈 분석 페이지 렌더링"""
    st.title("🔄 프랙탈 분석")
    st.divider()
    
    st.info("DTW 알고리즘을 이용한 패턴 매칭 기능은 5단계에서 구현됩니다.")
    
    # 분석 설정
    st.subheader("🎯 프랙탈 분석 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider(
            "패턴 윈도우 크기",
            min_value=20,
            max_value=100,
            value=config.FRACTAL_CONFIG['window_size'],
            step=10
        )
    
    with col2:
        min_similarity = st.slider(
            "최소 유사도 (%)",
            min_value=50,
            max_value=90,
            value=int(config.FRACTAL_CONFIG['min_similarity'] * 100),
            step=5
        )
    
    if st.button("🔍 프랙탈 분석 시작", type="primary", use_container_width=True):
        st.warning("⚠️ 프랙탈 분석 기능은 5단계에서 구현됩니다.")

# 차트 뷰어 페이지
def render_chart_viewer():
    """차트 뷰어 페이지 렌더링"""
    st.title("📉 차트 뷰어")
    st.divider()
    
    # 차트 설정
    col1, col2 = st.columns(2)
    with col1:
        chart_timeframe = st.selectbox(
            "타임프레임",
            options=list(config.TIMEFRAMES.keys()),
            format_func=lambda x: config.TIMEFRAMES[x],
            key="chart_tf"
        )
    
    with col2:
        chart_type = st.selectbox(
            "차트 타입",
            options=['캔들스틱', '라인'],
            key="chart_type"
        )
    
    # 데이터 로드
    collector = BTCDataCollector()
    df = collector.load_data(chart_timeframe)
    
    if df.empty:
        st.warning(f"⚠️ {config.TIMEFRAMES[chart_timeframe]} 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        
        # 샘플 차트 표시
        st.subheader("📊 BTC/USDT 차트 (샘플)")
        
        fig = go.Figure()
        
        # 더미 데이터로 샘플 차트 생성
        fig.add_trace(go.Candlestick(
            x=pd.date_range('2024-01-01', periods=100, freq='1h'),
            open=[40000 + i * 100 for i in range(100)],
            high=[40100 + i * 100 for i in range(100)],
            low=[39900 + i * 100 for i in range(100)],
            close=[40050 + i * 100 for i in range(100)],
            name='BTC/USDT'
        ))
        
        fig.update_layout(
            title='BTC/USDT 가격 차트 (샘플)',
            yaxis_title='가격 (USDT)',
            xaxis_title='시간',
            height=600,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # 기간 선택
    st.subheader("📅 기간 선택")
    col1, col2 = st.columns(2)
    
    with col1:
        # 빠른 선택 버튼
        period = st.radio(
            "빠른 선택",
            options=['전체', '1년', '6개월', '3개월', '1개월', '1주일'],
            horizontal=True
        )
    
    # 기간 계산
    end_date = df.index.max()
    if period == '전체':
        start_date = df.index.min()
    elif period == '1년':
        start_date = end_date - pd.Timedelta(days=365)
    elif period == '6개월':
        start_date = end_date - pd.Timedelta(days=180)
    elif period == '3개월':
        start_date = end_date - pd.Timedelta(days=90)
    elif period == '1개월':
        start_date = end_date - pd.Timedelta(days=30)
    else:  # 1주일
        start_date = end_date - pd.Timedelta(days=7)
    
    # 데이터 필터링
    df_filtered = df[df.index >= start_date]
    
    # 차트 생성
    st.subheader(f"📊 BTC/USDT {config.TIMEFRAMES[chart_timeframe]} 차트")
    
    fig = go.Figure()
    
    if chart_type == '캔들스틱':
        fig.add_trace(go.Candlestick(
            x=df_filtered.index,
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name='BTC/USDT'
        ))
    else:  # 라인 차트
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered['close'],
            mode='lines',
            name='종가',
            line=dict(color='#00D4FF', width=2)
        ))
    
    # 거래량 서브플롯 추가
    fig.add_trace(go.Bar(
        x=df_filtered.index,
        y=df_filtered['volume'],
        name='거래량',
        yaxis='y2',
        opacity=0.3,
        marker_color='gray'
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'BTC/USDT {config.TIMEFRAMES[chart_timeframe]} ({period})',
        yaxis=dict(
            title='가격 (USDT)',
            side='right'
        ),
        yaxis2=dict(
            title='거래량',
            overlaying='y',
            side='left',
            showgrid=False
        ),
        xaxis_title='시간',
        height=700,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # x축 범위 선택 버튼 추가
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 통계 정보
    st.divider()
    st.subheader("📈 통계 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df_filtered['close'].iloc[-1]
        price_change = df_filtered['close'].iloc[-1] - df_filtered['close'].iloc[0]
        price_change_pct = (price_change / df_filtered['close'].iloc[0]) * 100
        
        st.metric(
            "현재 가격",
            f"${current_price:,.2f}",
            f"{price_change:+,.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            "최고가",
            f"${df_filtered['high'].max():,.2f}",
            f"날짜: {df_filtered['high'].idxmax().strftime('%Y-%m-%d')}"
        )
    
    with col3:
        st.metric(
            "최저가",
            f"${df_filtered['low'].min():,.2f}",
            f"날짜: {df_filtered['low'].idxmin().strftime('%Y-%m-%d')}"
        )
    
    with col4:
        avg_volume = df_filtered['volume'].mean()
        st.metric(
            "평균 거래량",
            f"{avg_volume:,.0f}",
            f"총 {len(df_filtered):,} 캔들"
        )

# 메인 함수
def main():
    """메인 함수"""
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 현재 페이지에 따라 콘텐츠 렌더링
    if st.session_state.current_page == 'main':
        render_main_dashboard()
    elif st.session_state.current_page == 'data_collection':
        render_data_collection()
    elif st.session_state.current_page == 'indicator_analysis':
        render_indicator_analysis()
    elif st.session_state.current_page == 'fractal_analysis':
        render_fractal_analysis()
    elif st.session_state.current_page == 'chart_viewer':
        render_chart_viewer()

if __name__ == "__main__":
    main()
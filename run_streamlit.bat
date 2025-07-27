@echo off
echo BTC Analyzer Streamlit 앱을 시작합니다...
echo.

REM 가상환경 활성화
call venv\Scripts\activate.bat

REM Streamlit 앱 실행
streamlit run app.py

pause 
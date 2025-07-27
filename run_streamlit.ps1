Write-Host "BTC Analyzer Streamlit 앱을 시작합니다..." -ForegroundColor Green
Write-Host ""

# 가상환경 활성화
& ".\venv\Scripts\Activate.ps1"

# Streamlit 앱 실행
streamlit run app.py

Read-Host "엔터를 눌러 종료하세요" 
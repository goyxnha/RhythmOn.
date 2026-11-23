@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo [1/3] 가상환경을 만드는 중입니다...
  py -3.11 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo 가상환경을 활성화할 수 없습니다. Python 3.10 또는 3.11 설치를 확인하세요.
  pause
  exit /b 1
)

echo [2/3] 필요한 라이브러리를 설치합니다...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo 라이브러리 설치에 실패했습니다. 인터넷 연결과 Python 버전을 확인하세요.
  pause
  exit /b 1
)

echo [3/3] RhythmOn을 실행합니다...
python -m streamlit run app.py
pause

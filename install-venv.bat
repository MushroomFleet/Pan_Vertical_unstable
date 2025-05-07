@echo off
echo ===================================
echo Video Panner 3000 - Setup Script
echo ===================================
echo.

echo Creating Python virtual environment...
python -m venv venv
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Installing required packages...
pip install -r requirements.txt
echo.

echo ===================================
echo Virtual environment setup complete!
echo ===================================
echo.
echo To test the video panner, run:
echo   start-venv.bat
echo.
pause

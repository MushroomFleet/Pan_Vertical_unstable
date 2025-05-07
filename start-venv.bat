@echo off
echo ===================================
echo Video Panner 3000 - Test Runner
echo ===================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Running video panner test...
echo Command: python video_panner.py --config test-overlay.json --verbose --check-first-frame
echo.
python video_panner.py --config test.json --verbose --check-first-frame

echo.
echo ===================================
echo Test completed
echo ===================================
echo.
pause

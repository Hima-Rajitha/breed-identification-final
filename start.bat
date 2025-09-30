@echo off
echo ========================================
echo Cattle Breed Classification Web App
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Testing setup...
python test_setup.py

echo.
echo Starting the application...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python run_app.py

pause

@echo off
setlocal

set "ROOT=%~dp0"
set "APP_DIR=%ROOT%app"
set "INPUT=%ROOT%JIAxin WANG_create.xlsx"
set "OUTPUT=%ROOT%outputs\training_data_10000.csv"

cd /d "%APP_DIR%"

echo Generating 10000 slope safety training samples...
echo Input:  %INPUT%
echo Output: %OUTPUT%
echo.

conda run -n pytorch_wjx python src\generate_training_data.py --input "%INPUT%" --output "%OUTPUT%" --n-samples 10000 --num-slices 30 --search-density 8 --search-mode standard --progress-interval 50

if errorlevel 1 (
    echo.
    echo Failed to generate training data.
    pause
    exit /b 1
)

echo.
echo Done. Training data saved to:
echo %OUTPUT%
pause

endlocal

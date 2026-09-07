@echo off
setlocal

set "ROOT=%~dp0"
set "APP_DIR=%ROOT%app"
set "PORT=8501"

cd /d "%APP_DIR%"

echo Starting Slope Safety App on http://localhost:%PORT%
echo Close this window to stop the local app.
echo.

start "" "http://localhost:%PORT%"
conda run -n pytorch_wjx streamlit run app.py --server.headless=true --server.port=%PORT% --browser.gatherUsageStats=false

endlocal


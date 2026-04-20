@echo off
:: ================================================================
::  run_all.bat — Training + App dalam satu klik
:: ================================================================

title HemoScan - Training & App
color 0F

echo.
echo  ============================================================
echo   HEMOSCAN — DETEKSI ANEMIA DARI KONJUNGTIVA
echo   Run Training + App
echo  ============================================================
echo.

cd /d "%~dp0"

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python tidak ditemukan.
    pause
    exit /b 1
)

echo  [1/2] Checking model files...

if not exist "model_output\pca.pkl" (
    echo  [INFO] Model files tidak lengkap - menjalankan training...
    echo.
    python prediction_engine.py
    if errorlevel 1 (
        echo  [ERROR] Training gagal!
        pause
        exit /b 1
    )
) else (
    echo  [OK] Model files sudah ada!
)

echo.
echo  [2/2] Launching Streamlit app...
echo.
echo  ============================================================
echo   App URL: http://localhost:8501
echo   Press Ctrl+C untuk keluar
echo  ============================================================
echo.

python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause

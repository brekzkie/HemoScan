@echo off
:: ================================================================
::  run_app.bat
::  Jalankan aplikasi Streamlit deteksi anemia
::  Pastikan training sudah selesai dulu (run_training.bat)
:: ================================================================

title HemoScan — Aplikasi Deteksi Anemia
color 0B

echo.
echo  ============================================================
echo   HEMOSCAN — DETEKSI ANEMIA DARI KONJUNGTIVA
echo   Streamlit Application
echo  ============================================================
echo.

cd /d "%~dp0"

:: ── Cek model sudah ada ─────────────────────────────────────────
if not exist "model_output\pca.pkl" (
    echo  [ERROR] Model belum ada!
    echo.
    echo  Jalankan dulu: run_training.bat
    echo  Tunggu sampai training selesai, baru buka aplikasi ini.
    echo.
    pause
    exit /b 1
)

if not exist "model_output\xgb_anemia.json" (
    echo  [ERROR] File xgb_anemia.json tidak ditemukan!
    echo  Jalankan ulang run_training.bat
    pause
    exit /b 1
)

if not exist "model_output\config.pkl" (
    echo  [ERROR] File config.pkl tidak ditemukan!
    echo  Jalankan ulang run_training.bat
    pause
    exit /b 1
)

echo  [OK] Model ditemukan di model_output\
echo.

:: ── Install streamlit jika belum ───────────────────────────────
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Menginstall Streamlit...
    pip install streamlit --quiet
)

:: ── Jalankan app ────────────────────────────────────────────────
echo  [INFO] Membuka aplikasi di browser...
echo  (Tekan Ctrl+C di window ini untuk menutup aplikasi)
echo.
echo  URL: http://localhost:8501
echo.

streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
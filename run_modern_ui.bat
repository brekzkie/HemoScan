@echo off
title HemoScan Modern UI — Auto Fix
color 0E

echo.
echo  ============================================================
echo   HEMOSCAN — MODERN WEB INTERFACE (AUTO FIX)
echo  ============================================================
echo.

cd /d "%~dp0"

:: ── Step 1: Install Dependencies ─────────────────────────────
echo  [1/3] Memeriksa dan menginstal pustaka yang diperlukan...
echo  (Ini mungkin memakan waktu beberapa menit jika baru pertama kali)
pip install -r requirements.txt --quiet
pip install python-multipart uvicorn fastapi --quiet

:: ── Step 2: Cek Model ───────────────────────────────────────
echo  [2/3] Memverifikasi file AI Model...
if not exist "model_output\pca.pkl" (
    echo  [ERROR] Folder model_output atau isinya belum ada!
    echo  Silakan jalankan run_training.bat terlebih dahulu.
    pause
    exit /b 1
)

:: ── Step 3: Jalankan Server ──────────────────────────────────
echo  [3/3] Menjalankan server...
echo.
echo  ============================================================
echo   SERVER BERHASIL DIJALANKAN!
echo   Buka browser dan ketik: http://localhost:8000
echo  ============================================================
echo.
echo  (Tekan CTRL+C untuk mematikan server)
echo.

python server.py
pause

@echo off
:: ================================================================
::  run_training.bat
::  Jalankan training pipeline deteksi anemia
::  Klik dua kali file ini, atau jalankan dari Command Prompt
:: ================================================================

title HemoScan — Training Pipeline
color 0A

echo.
echo  ============================================================
echo   HEMOSCAN — DETEKSI ANEMIA DARI KONJUNGTIVA
echo   Training Pipeline
echo  ============================================================
echo.

:: ── Cek Python tersedia ─────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python tidak ditemukan.
    echo  Install dari https://python.org dan pastikan ada di PATH.
    pause
    exit /b 1
)

echo  [OK] Python ditemukan:
python --version
echo.

:: ── Pindah ke folder script ini ────────────────────────────────
cd /d "%~dp0"
echo  [INFO] Working directory: %CD%
echo.

:: ── Install dependencies ────────────────────────────────────────
echo  [STEP 1/2] Menginstall dependencies...
echo  (Proses ini hanya perlu dilakukan sekali)
echo.
pip install tensorflow xgboost scikit-learn pandas numpy openpyxl opencv-python pillow joblib matplotlib seaborn scipy --quiet
if errorlevel 1 (
    echo.
    echo  [ERROR] Gagal install dependencies.
    echo  Coba jalankan: pip install -r requirements_training.txt
    pause
    exit /b 1
)
echo.
echo  [OK] Dependencies siap.
echo.

:: ── Jalankan training ───────────────────────────────────────────
echo  [STEP 2/2] Memulai training...
echo  ============================================================
echo  Estimasi waktu:
echo    - Ekstraksi gambar  : 5-15 menit (tergantung jumlah gambar)
echo    - Training XGBoost  : 2-5 menit
echo    - Cross-validation  : 5-15 menit
echo  ============================================================
echo.

python prediction_engine.py

:: ── Cek hasil ───────────────────────────────────────────────────
if errorlevel 1 (
    echo.
    echo  [ERROR] Training gagal. Lihat pesan error di atas.
    echo  Periksa:
    echo    1. Path dataset di CELL 2 sudah benar?
    echo    2. File India.xlsx dan Italy.xlsx ada?
    echo    3. Folder gambar sudah berisi file .png?
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo  [SELESAI] Training berhasil!
echo  File model tersimpan di folder: model_output\
echo.
echo  File yang dihasilkan:
echo    - model_output\pca.pkl
echo    - model_output\config.pkl
echo    - model_output\xgb_anemia.json
echo    - model_output\evaluation_full.png
echo    - model_output\scree_plot.png
echo  ============================================================
echo.
echo  Langkah selanjutnya: jalankan run_app.bat untuk membuka aplikasi
echo.
pause
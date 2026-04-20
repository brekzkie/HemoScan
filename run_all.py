#!/usr/bin/env python
"""
Script para menjalankan training (jika diperlukan) dan app Streamlit
"""
import os
import sys
import subprocess

def main():
    print("\n" + "="*60)
    print("HEMOSCAN - DETEKSI ANEMIA DARI KONJUNGTIVA")
    print("="*60 + "\n")

    # Cek model files
    model_dir = "model_output"
    required_files = ['pca.pkl', 'xgb_anemia.json', 'config.pkl']

    print("[1] Cek model files...")
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

    if missing:
        print(f"    ❌ Files hilang: {missing}")
        print(f"\n[2] Jalankan training pipeline...")
        result = subprocess.run([sys.executable, 'prediction_engine.py'], cwd=os.getcwd())
        if result.returncode != 0:
            print("    ❌ Training gagal!")
            sys.exit(1)
        print("    ✅ Training selesai!")
    else:
        print("    ✅ Semua model files ada!")

    # Jalankan streamlit app
    print(f"\n[3] Launching Streamlit app...\n")
    print("-" * 60)
    print("🔗 App akan dibuka di: http://localhost:8501")
    print("📝 Tekan Ctrl+C untuk menutup aplikasi")
    print("-" * 60 + "\n")

    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', '8501',
        '--browser.gatherUsageStats', 'false'
    ])

if __name__ == '__main__':
    main()

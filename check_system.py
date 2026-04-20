#!/usr/bin/env python3
"""
Script untuk verify & test HemoScan components
"""
import os
import sys

print("\n" + "="*60)
print("🔍 HEMOSCAN - SYSTEM CHECK")
print("="*60 + "\n")

# Check 1: Model Files
print("[CHECK 1] Model Files")
print("-" * 40)
model_dir = "model_output"
required_files = {
    'pca.pkl': 'PCA dimensionality reducer',
    'xgb_anemia.json': 'XGBoost classifier',
    'config.pkl': 'Model configuration',
    'label_encoder_status.pkl': 'Label encoder'
}

all_present = True
for filename, description in required_files.items():
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024  # KB
        print(f"  ✅ {filename:<30} ({size:>6.1f} KB) - {description}")
    else:
        print(f"  ❌ {filename:<30} - MISSING")
        all_present = False

# Check 2: Python Dependencies
print("\n[CHECK 2] Python Dependencies")
print("-" * 40)
deps = ['tensorflow', 'xgboost', 'streamlit', 'pandas', 'sklearn', 'cv2', 'PIL', 'numpy']
for dep in deps:
    try:
        if dep == 'sklearn':
            import sklearn
        elif dep == 'cv2':
            import cv2
        elif dep == 'PIL':
            from PIL import Image
        else:
            __import__(dep)
        print(f"  ✅ {dep:<20} installed")
    except ImportError:
        print(f"  ❌ {dep:<20} NOT installed")

# Check 3: Dataset
print("\n[CHECK 3] Dataset")
print("-" * 40)
datasets = {
    'dataset anemia/India': 'India images',
    'dataset anemia/Italy': 'Italy images',
}
for folder, desc in datasets.items():
    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"  ✅ {folder:<30} ({len(files)} items) - {desc}")
    else:
        print(f"  ❌ {folder:<30} - NOT FOUND")

# Check 4: App Files
print("\n[CHECK 4] Application Files")
print("-" * 40)
app_files = ['app.py', 'inference.py', 'requirements.txt']
for fname in app_files:
    if os.path.exists(fname):
        size = os.path.getsize(fname) / 1024
        print(f"  ✅ {fname:<20} ({size:>6.1f} KB)")
    else:
        print(f"  ❌ {fname:<20} - NOT FOUND")

# Summary
print("\n" + "="*60)
if all_present:
    print("✅ SYSTEM READY - Jalankan:")
    print("   python run_all.py")
    print("   atau")
    print("   double-click run_all.bat")
else:
    print("⚠️  SYSTEM NOT READY - Perlu training:")
    print("   python prediction_engine.py")

print("="*60 + "\n")

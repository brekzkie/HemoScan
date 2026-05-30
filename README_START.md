# 🚀 HEMOSCAN - QUICK START GUIDE

<img width="3840" height="3840" alt="image" src="https://github.com/user-attachments/assets/650e0941-d8bb-4ad3-8754-f010a91327aa" />


## Status Sistem ✅

Model files sudah ready:
- ✅ `model_output/pca.pkl` - PCA reducer (181 KB)
- ✅ `model_output/xgb_anemia.json` - XGBoost classifier (79 KB)
- ✅ `model_output/config.pkl` - Configuration (330 B)
- ✅ `model_output/label_encoder_status.pkl` - Label encoder (494 B)

Dataset tersedia:
- ✅ `dataset anemia/India/` - India konjungtiva images
- ✅ `dataset anemia/Italy/` - Italy konjungtiva images
- ✅ `metadata_anemia_combined.csv` - Patient metadata

---

## 🎯 Cara Menjalankan

### **CARA 1: Double-click batch file (Termudah)**

1. **Windows Explorer** → navigasi ke folder `HemoScan`
2. **Double-click `run_all.bat`**
3. Tunggu ~1-2 menit untuk dependencies loading
4. Browser akan otomatis buka ke `http://localhost:8501`

---

### **CARA 2: Command Line**

**Terminal / PowerShell:**
```bash
cd "c:\NANDA\TUGAS\SEMESTER 6\Proyek Sains Data\HemoScan"

# Opsi A: Jalankan langsung Streamlit app (model sudah ada)
streamlit run app.py

# Opsi B: Jalankan training dulu (baru kalau ingin retrain)
python prediction_engine.py

# Opsi C: Jalankan semua (training + app)
python run_all.py
```

---

### **CARA 3: Batch scripts one-by-one**

**Start Training:**
```batch
double-click run_training.bat
```
(Tunggu sampai training selesai - estimasi 30 menit pertama kali)

**Start App (setelah training selesai):**
```batch
double-click run_app.bat
```

---

## 🏗️ Struktur Aplikasi

```
STREAMLIT UI (app.py)
    ├── 📧 Login Panel
    ├── 📋 Patient Form (Age, Gender, Hemoglobin, etc)
    ├── 🖼️  Image Upload (Konjungtiva)
    └── 📊 Results & Visualization

INFERENCE ENGINE (inference.py)
    ├── Load EfficientNetB0 (feature extractor)
    ├── Apply PCA (dimensionality reduction)
    ├── Run XGBoost (anemia classifier)
    └── Return prediction + confidence

TRAINING PIPELINE (prediction_engine.py)
    ├── Extract & validate images from dataset
    ├── Load patient metadata
    ├── Extract features using EfficientNetB0
    ├── Apply PCA
    ├── Train & evaluate XGBoost
    └── Save models to model_output/
```

---

## 📝 Fitur Aplikasi

✅ **Upload Gambar Konjungtiva** - Prediksi anemia dari visual cues
✅ **Patient Info Integration** - Age, gender, hemoglobin data
✅ **Model Prediction** - Confidence scores & classification
✅ **Result Visualization** - Charts & medical insights

---

## ⚙️ Technical Stack

| Component | Version |
|-----------|---------|
| **TensorFlow** | ≥ 2.13.0 |
| **XGBoost** | ≥ 2.0.0 |
| **Streamlit** | ≥ 1.35.0 |
| **scikit-learn** | ≥ 1.3.0 |
| **pandas** | ≥ 2.0.0 |
| **numpy** | ≥ 1.24.0 |

---

## 🔧 Troubleshooting

### Browser tidak buka otomatis?
→ Manual navigate ke `http://localhost:8501`

### "ModuleNotFoundError: No module named 'tensorflow'"?
→ Run: `pip install -r requirements.txt`

### App crash dengan "Model file not found"?
→ Jalankan `run_training.bat` atau `python prediction_engine.py` dulu

### Port 8501 sudah terpakai?
→ Streamlit akan auto-use port berikutnya, atau specify:
```bash
streamlit run app.py --server.port 8502
```

---

## 📌 Files Summary

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit UI |
| `inference.py` | Prediction engine |
| `prediction_engine.py` | Training pipeline |
| `run_all.bat` | Run training + app (NEW) |
| `run_training.bat` | Run training only |
| `run_app.bat` | Run app only |
| `check_system.py` | System verification (NEW) |

---

## ✨ Next Steps

1. **Run `run_all.bat`** → System starts training (if needed) + launches app
2. **Upload image** → System predicts anemia status
3. **View results** → Confidence score + medical insights

**Estimated time to first prediction: 2-5 minutes**

---

Generated: 2026-04-13
HemoScan v1.0 - Anemia Detection System

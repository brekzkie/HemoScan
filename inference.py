# ================================================================
#  inference.py
#  Engine prediksi anemia — dipakai oleh app.py (Streamlit)
#  Pastikan file model sudah ada di MODEL_DIR sebelum menjalankan app
# ================================================================

import os
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ── Path default model (sesuaikan jika berbeda) ────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_output')


# ================================================================
# Helper: hitung age_group dari umur
# ================================================================
def hitung_age_group(age: int) -> str:
    if age < 25:
        return 'Anak-anak dan Remaja'
    elif age < 60:
        return 'Dewasa'
    return 'Lansia'


# ================================================================
# Load semua komponen model (dipanggil sekali, di-cache Streamlit)
# ================================================================
def load_model_components(model_dir: str = MODEL_DIR):
    """
    Load EfficientNetB0, PCA, XGBoost, dan config dari folder model.

    Returns
    -------
    dict dengan kunci: base_model, pca, xgb_model, config
    """
    # 1. Validasi folder
    required = ['pca.pkl', 'config.pkl', 'xgb_anemia.json']
    missing  = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"File model tidak ditemukan di '{model_dir}': {missing}\n"
            "Jalankan dulu pipeline training (anemia_detection_final.py) di Kaggle."
        )

    # 2. Load komponen
    config    = joblib.load(os.path.join(model_dir, 'config.pkl'))
    pca       = joblib.load(os.path.join(model_dir, 'pca.pkl'))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(model_dir, 'xgb_anemia.json'))

    # 3. Rebuild EfficientNetB0 (weights hanya dari ImageNet, tidak perlu file tambahan)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # inference mode — tidak perlu gradient

    return {
        'base_model': base_model,
        'pca'       : pca,
        'xgb_model' : xgb_model,
        'config'    : config,
    }


# ================================================================
# Fungsi utama: prediksi dari PIL Image + data pasien
# ================================================================
def predict_from_pil(
    pil_image : Image.Image,
    nama      : str,
    umur      : int,
    gender    : str,          # 'M' atau 'F'
    components: dict,
) -> dict:
    """
    Prediksi status anemia dari objek PIL Image + data pasien.

    Parameters
    ----------
    pil_image  : PIL.Image  — gambar konjungtiva yang sudah diupload
    nama       : str        — nama pasien (hanya untuk output)
    umur       : int        — usia dalam tahun
    gender     : str        — 'M' (Laki-laki) atau 'F' (Perempuan)
    components : dict       — hasil load_model_components()

    Returns
    -------
    dict:
        nama, umur, gender, age_group,
        prediksi ('Anemia' | 'Non-Anemia'),
        probabilitas (float 0–1),
        threshold (float),
        keterangan (str),
        risk_level ('high' | 'medium' | 'low')
    """
    base_model    = components['base_model']
    pca           = components['pca']
    xgb_model     = components['xgb_model']
    config        = components['config']

    age_map       = config['age_map']
    gender_map    = config['gender_map']
    best_thresh   = config['best_thresh']
    anemia_class  = config['anemia_class']
    age_group     = hitung_age_group(umur)

    # ── 1. Preprocessing gambar ────────────────────────────────
    img_rgb  = pil_image.convert('RGB').resize((224, 224))
    arr      = keras_image.img_to_array(img_rgb)
    arr      = preprocess_input(np.expand_dims(arr, axis=0))

    # ── 2. Ekstraksi fitur EfficientNetB0 ──────────────────────
    img_feat = base_model.predict(arr, verbose=0).flatten()   # (1280,)

    # ── 3. PCA ─────────────────────────────────────────────────
    img_pca  = pca.transform(img_feat.reshape(1, -1))          # (1, N_PCA)

    # ── 4. Encode fitur tabular ────────────────────────────────
    age_enc  = age_map.get(age_group, 1)
    gen_enc  = gender_map.get(gender, 0)
    tabular  = np.array([[umur, gen_enc, age_enc]], dtype='float32')  # (1, 3)

    # ── 5. Gabungkan & prediksi ────────────────────────────────
    X_input  = np.concatenate([tabular, img_pca], axis=1)
    proba_arr = xgb_model.predict_proba(X_input)[0]
    proba    = float(proba_arr[anemia_class])
    prediksi = 'Anemia' if proba >= best_thresh else 'Non-Anemia'

    # ── 6. Risk level untuk UI ─────────────────────────────────
    if prediksi == 'Anemia':
        if proba >= 0.80:
            risk_level = 'high'
        else:
            risk_level = 'medium'
    else:
        risk_level = 'low'

    # ── 7. Keterangan kontekstual ──────────────────────────────
    if prediksi == 'Anemia':
        keterangan = (
            f"Berdasarkan analisis gambar konjungtiva dan data pasien, "
            f"{nama} terindikasi mengalami anemia (probabilitas {proba:.1%}). "
            f"Disarankan untuk segera melakukan pemeriksaan darah lengkap "
            f"dan berkonsultasi dengan dokter."
        )
    else:
        keterangan = (
            f"Berdasarkan analisis gambar konjungtiva dan data pasien, "
            f"{nama} tidak terindikasi anemia (probabilitas anemia {proba:.1%}). "
            f"Tetap jaga pola makan bergizi seimbang dan lakukan pemeriksaan "
            f"rutin jika ada keluhan."
        )

    return {
        'nama'        : nama,
        'umur'        : umur,
        'gender'      : 'Laki-laki' if gender == 'M' else 'Perempuan',
        'age_group'   : age_group,
        'prediksi'    : prediksi,
        'probabilitas': round(proba, 4),
        'threshold'   : best_thresh,
        'keterangan'  : keterangan,
        'risk_level'  : risk_level,
    }
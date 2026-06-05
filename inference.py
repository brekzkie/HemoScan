# ================================================================
#  inference.py
#  Engine prediksi anemia — menggunakan ONNX Runtime & OpenCV
# ================================================================

import os
import numpy as np
import joblib
import cv2
import onnxruntime as ort
import random
from PIL import Image

# ── Path default model ────────────────
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
# Image Preprocessing & Augmentasi (OpenCV / NumPy)
# ================================================================

def preprocess_image_eval(img_rgb: np.ndarray) -> np.ndarray:
    """
    Ekuivalen dengan aug_eval di notebook:
    - Resize mempertahankan rasio aspek sehingga sisi terpanjang = 224.
    - Pad dengan border hitam (value 0) hingga ukuran 224x224.
    """
    h, w = img_rgb.shape[:2]
    scale = 224.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_y = 224 - new_h
    pad_x = 224 - new_w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return img_padded


def apply_aug_tta(img_rgb: np.ndarray) -> np.ndarray:
    """
    Ekuivalen dengan aug_tta di notebook:
    - Melakukan flip horizontal (p=0.5).
    - Melakukan ShiftScaleRotate (p=0.5) dengan limit kecil.
    - Mengatur Brightness & Contrast secara random (p=0.5).
    - Menghasilkan gambar ukuran 224x224.
    """
    # Awali dengan preprocessing eval agar ukuran pas 224x224
    img = preprocess_image_eval(img_rgb)
    
    # 1. Horizontal Flip (p=0.5)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        
    # 2. ShiftScaleRotate (p=0.5)
    if random.random() < 0.5:
        h, w = img.shape[:2]
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.95, 1.05)
        tx = random.uniform(-0.03 * w, 0.03 * w)
        ty = random.uniform(-0.03 * h, 0.03 * h)
        
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
    # 3. RandomBrightnessContrast (p=0.5)
    if random.random() < 0.5:
        alpha = random.uniform(0.90, 1.10) # Kontras
        beta = random.uniform(-25.5, 25.5) # Kecerahan (0.10 * 255)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    return img


# ================================================================
# Load semua komponen model
# ================================================================
def load_model_components(model_dir: str = MODEL_DIR):
    """
    Load model ONNX dan konfigurasinya.
    """
    onnx_path = os.path.join(model_dir, 'anemia_model_final.onnx')
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Model ONNX tidak ditemukan di '{onnx_path}'")

    session = ort.InferenceSession(onnx_path)

    # Coba load config.pkl jika ada
    config_path = os.path.join(model_dir, 'config.pkl')
    if os.path.exists(config_path):
        try:
            config = joblib.load(config_path)
        except Exception:
            config = {}
    else:
        config = {}

    # Override/set default parameter dari model baru
    config['best_thresh'] = 0.50
    config['n_tta'] = 8
    config['anemia_idx'] = 0
    config['classes'] = ['Anemia', 'Non-Anemia']

    return {
        'onnx_session': session,
        'config': config,
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
    Prediksi status anemia menggunakan model ONNX end-to-end + TTA.
    """
    session = components['onnx_session']
    config  = components['config']

    best_thresh = config.get('best_thresh', 0.50)
    n_tta       = config.get('n_tta', 8)
    age_group   = hitung_age_group(umur)

    # Konversi PIL Image ke numpy RGB
    img_rgb = np.array(pil_image.convert('RGB'))

    probas = []
    for i in range(n_tta):
        if i == 0:
            aug_img = preprocess_image_eval(img_rgb)
        else:
            aug_img = apply_aug_tta(img_rgb)

        arr = aug_img.astype(np.float32)
        arr = np.expand_dims(arr, axis=0) # (1, 224, 224, 3)

        # Jalankan prediksi ONNX
        outputs = session.run(["output"], {"input_layer": arr})
        proba_raw = float(outputs[0][0, 0])
        probas.append(proba_raw)

    # Rata-rata probabilitas TTA
    mean_proba = float(np.mean(probas))

    # Ground truth mapping:
    # index 0: Anemia, index 1: Non-Anemia
    # Output sigmoid Keras memprediksi probabilitas class 1 (Non-Anemia)
    # Probabilitas Anemia = 1.0 - mean_proba
    proba_anemia = 1.0 - mean_proba

    # Klasifikasi
    prediksi = 'Anemia' if proba_anemia >= best_thresh else 'Non-Anemia'

    # Risk level untuk UI
    if prediksi == 'Anemia':
        if proba_anemia >= 0.80:
            risk_level = 'high'
        else:
            risk_level = 'medium'
    else:
        risk_level = 'low'

    # Keterangan kontekstual
    if prediksi == 'Anemia':
        keterangan = (
            f"Berdasarkan analisis gambar konjungtiva dan data pasien, "
            f"{nama} terindikasi mengalami anemia (probabilitas {proba_anemia:.1%}). "
            f"Disarankan untuk segera melakukan pemeriksaan darah lengkap "
            f"dan berkonsultasi dengan dokter."
        )
    else:
        keterangan = (
            f"Berdasarkan analisis gambar konjungtiva dan data pasien, "
            f"{nama} tidak terindikasi anemia (probabilitas anemia {proba_anemia:.1%}). "
            f"Tetap jaga pola makan bergizi seimbang dan lakukan pemeriksaan "
            f"rutin jika ada keluhan."
        )

    return {
        'nama'        : nama,
        'umur'        : umur,
        'gender'      : 'Laki-laki' if gender == 'M' else 'Perempuan',
        'age_group'   : age_group,
        'prediksi'    : prediksi,
        'probabilitas': round(proba_anemia, 4),
        'threshold'   : best_thresh,
        'keterangan'  : keterangan,
        'risk_level'  : risk_level,
    }
# ================================================================
#  inference.py
#  Engine prediksi anemia — ONNX Runtime & OpenCV
# ================================================================

import os
import random

import cv2
import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image

from config import MODEL_DIR, DEFAULT_THRESHOLD, DEFAULT_N_TTA


# ================================================================
# Helper: hitung kelompok usia
# ================================================================
def hitung_age_group(age: int) -> str:
    """Mapping umur ke kelompok usia untuk konteks keterangan."""
    if age < 25:
        return "Anak-anak dan Remaja"
    elif age < 60:
        return "Dewasa"
    return "Lansia"


# ================================================================
# Image Preprocessing & Test-Time Augmentation (TTA)
# ================================================================

def preprocess_image_eval(img_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocessing evaluasi: resize mempertahankan aspek rasio (sisi
    terpanjang = 224), lalu pad dengan border hitam hingga 224×224.
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

    return cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0,
    )


def apply_aug_tta(img_rgb: np.ndarray) -> np.ndarray:
    """
    Test-Time Augmentation (TTA):
    - Flip horizontal (p=0.5)
    - ShiftScaleRotate (p=0.5) dengan limit kecil
    - RandomBrightnessContrast (p=0.5)
    Menghasilkan gambar 224×224.
    """
    img = preprocess_image_eval(img_rgb)

    # Horizontal Flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # ShiftScaleRotate
    if random.random() < 0.5:
        h, w = img.shape[:2]
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.95, 1.05)
        tx = random.uniform(-0.03 * w, 0.03 * w)
        ty = random.uniform(-0.03 * h, 0.03 * h)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # RandomBrightnessContrast
    if random.random() < 0.5:
        alpha = random.uniform(0.90, 1.10)  # kontras
        beta = random.uniform(-25.5, 25.5)  # kecerahan
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img


# ================================================================
# Load komponen model
# ================================================================
def load_model_components(model_dir: str = MODEL_DIR) -> dict:
    """
    Load model ONNX dan konfigurasinya dari model_dir.

    Returns:
        dict dengan key 'onnx_session' dan 'config'.
    """
    onnx_path = os.path.join(model_dir, "anemia_model_final.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Model ONNX tidak ditemukan: '{onnx_path}'")

    session = ort.InferenceSession(onnx_path)

    # Load config.pkl jika ada
    config_path = os.path.join(model_dir, "config.pkl")
    config: dict = {}
    if os.path.exists(config_path):
        try:
            config = joblib.load(config_path)
        except Exception:
            config = {}

    # Override parameter untuk pipeline baru
    config["best_thresh"] = DEFAULT_THRESHOLD
    config["n_tta"] = DEFAULT_N_TTA
    config["anemia_idx"] = 0
    config["classes"] = ["Anemia", "Non-Anemia"]

    return {
        "onnx_session": session,
        "config": config,
    }


# ================================================================
# Prediksi dari PIL Image + data pasien
# ================================================================
def predict_from_pil(
    pil_image: Image.Image,
    nama: str,
    umur: int,
    gender: str,
    components: dict,
) -> dict:
    """
    Prediksi status anemia menggunakan model ONNX end-to-end dengan TTA.

    Args:
        pil_image: Gambar konjungtiva (sudah di-crop).
        nama: Nama pasien.
        umur: Usia pasien.
        gender: 'M' (laki-laki) atau 'F' (perempuan).
        components: Dict dari load_model_components().

    Returns:
        Dict berisi nama, umur, gender, age_group, prediksi,
        probabilitas, threshold, keterangan, dan risk_level.
    """
    session = components["onnx_session"]
    config = components["config"]

    best_thresh = config.get("best_thresh", DEFAULT_THRESHOLD)
    n_tta = config.get("n_tta", DEFAULT_N_TTA)
    age_group = hitung_age_group(umur)

    # Konversi PIL → numpy RGB
    img_rgb = np.array(pil_image.convert("RGB"))

    # TTA: n_tta forward passes
    probas: list[float] = []
    for i in range(n_tta):
        aug_img = preprocess_image_eval(img_rgb) if i == 0 else apply_aug_tta(img_rgb)

        arr = aug_img.astype(np.float32)
        arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

        outputs = session.run(["output"], {"input_layer": arr})
        proba_raw = float(outputs[0][0, 0])
        probas.append(proba_raw)

    # Rata-rata probabilitas TTA
    mean_proba = float(np.mean(probas))

    # Output sigmoid → probabilitas class "Non-Anemia"
    # Maka probabilitas Anemia = 1.0 - mean_proba
    proba_anemia = 1.0 - mean_proba

    # Klasifikasi
    prediksi = "Anemia" if proba_anemia >= best_thresh else "Non-Anemia"

    # Risk level
    if prediksi == "Anemia":
        risk_level = "high" if proba_anemia >= 0.80 else "medium"
    else:
        risk_level = "low"

    # Keterangan kontekstual
    if prediksi == "Anemia":
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
        "nama": nama,
        "umur": umur,
        "gender": "Laki-laki" if gender == "M" else "Perempuan",
        "age_group": age_group,
        "prediksi": prediksi,
        "probabilitas": round(proba_anemia, 4),
        "threshold": best_thresh,
        "keterangan": keterangan,
        "risk_level": risk_level,
    }
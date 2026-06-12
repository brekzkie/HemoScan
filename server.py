# ================================================================
#  server.py
#  FastAPI backend server untuk HemoScan
# ================================================================

import os
import uuid
import datetime
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from psycopg2.extras import RealDictCursor
import numpy as np
import io

from config import get_db_connection, UPLOADS_DIR
from inference import load_model_components, predict_from_pil
from augmentation import flip_left_to_right
from image_processing import validate_conjunctiva_image, crop_conjunctiva


# ─── Pydantic Models ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str


class LinkScansRequest(BaseModel):
    email: str
    scan_ids: list[str]


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(title="HemoScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model components saat startup
try:
    COMPONENTS = load_model_components()
    print("[Model] Components loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    COMPONENTS = None

# Pastikan folder uploads ada
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ─── Helper: Preprocessing Info ──────────────────────────────────
def get_preprocessing_info() -> dict:
    """Metadata pipeline ML untuk ditampilkan di UI."""
    if COMPONENTS is None:
        return {}
    config = COMPONENTS.get("config", {})
    best_thresh = config.get("best_thresh", 0.5)
    return {
        "img_input_size": [224, 224, 3],
        "backbone": "EfficientNetB0 (ONNX)",
        "img_feat_dim": 1280,
        "pca_components": 0,
        "tabular_features": [],
        "age_map": {},
        "gender_map": {},
        "classifier": "Dense Head (Sigmoid)",
        "best_threshold": best_thresh,
    }


# ─── Register Endpoint ───────────────────────────────────────────
@app.post("/api/register")
async def register(req: RegisterRequest):
    if not req.email or not req.password or not req.display_name:
        raise HTTPException(status_code=400, detail="Semua kolom wajib diisi")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password minimal 6 karakter")
    if len(req.display_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Nama lengkap minimal 2 karakter")

    import bcrypt

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id FROM public.users WHERE email = %s", (req.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=409, detail="Email sudah terdaftar")

            password_hash = bcrypt.hashpw(
                req.password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")
            cursor.execute(
                """
                INSERT INTO public.users (email, password_hash, display_name, role)
                VALUES (%s, %s, %s, 'user')
                """,
                (req.email, password_hash, req.display_name.strip()),
            )
            conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "status": "success",
        "message": "Pendaftaran berhasil! Silakan masuk.",
        "email": req.email,
        "display_name": req.display_name.strip(),
    }


# ─── Login Endpoint ───────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    import bcrypt

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM public.users WHERE email = %s", (req.email,))
            user = cursor.fetchone()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user:
        raise HTTPException(status_code=401, detail="Email atau password salah")

    if bcrypt.checkpw(req.password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return {
            "status": "success",
            "message": "Login successful",
            "email": user["email"],
            "role": user["role"],
            "display_name": user["display_name"],
        }
    raise HTTPException(status_code=401, detail="Email atau password salah")


# ─── Link Scans Endpoint ──────────────────────────────────────────
@app.post("/api/link-scans")
async def link_scans(req: LinkScansRequest):
    if not req.email or not req.scan_ids:
        return {"status": "success", "message": "No scans to link"}

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            for scan_id in req.scan_ids:
                cursor.execute(
                    """
                    UPDATE public.scans
                    SET user_email = %s
                    WHERE scan_id = %s AND (user_email = '' OR user_email IS NULL)
                    """,
                    (req.email, scan_id),
                )
            conn.commit()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Database error while linking scans: {str(e)}"
        )

    return {"status": "success", "message": f"Linked {len(req.scan_ids)} scans to {req.email}"}


# ─── Preprocessing Info Endpoint ─────────────────────────────────
@app.get("/api/preprocessing-info")
async def preprocessing_info():
    """Return konfigurasi pipeline ML untuk UI."""
    return get_preprocessing_info()


# ─── Predict Endpoint ─────────────────────────────────────────────
@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    symptoms: str = Form(""),
    user_email: str = Form(""),
    eye_side: str = Form("right"),
):
    if COMPONENTS is None:
        raise HTTPException(status_code=500, detail="Model components not loaded")

    # Read & validate image
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    is_valid, validation_msg = validate_conjunctiva_image(pil_image)
    if not is_valid:
        raise HTTPException(status_code=400, detail=validation_msg)

    # Flip jika sisi kiri
    if eye_side == "left":
        img_np = np.array(pil_image)
        flipped_np = flip_left_to_right(img_np)
        pil_image = Image.fromarray(flipped_np)

    # Simpan gambar
    image_id = f"{uuid.uuid4()}.png"
    image_path = os.path.join(UPLOADS_DIR, image_id)
    pil_image.save(image_path)

    # Crop konjungtiva
    try:
        cropped_image = crop_conjunctiva(pil_image)
        cropped_image_id = f"cropped_{image_id}"
        cropped_image_path = os.path.join(UPLOADS_DIR, cropped_image_id)
        cropped_image.save(cropped_image_path)
        cropped_image_url = f"/uploads/{cropped_image_id}"
    except Exception as e:
        print(f"Error cropping conjunctiva: {e}")
        cropped_image_url = f"/uploads/{image_id}"
        cropped_image = pil_image

    # Prediksi
    result = predict_from_pil(cropped_image, name, age, gender, COMPONENTS)

    # Metadata
    scan_id = f"#SCN-{uuid.uuid4().hex[:4].upper()}"
    timestamp = datetime.datetime.now().strftime("%b %d, %Y • %H:%M")
    symptoms_list = symptoms.split(",") if symptoms else []

    # Detail preprocessing untuk UI
    steps_list = [
        {
            "name": "Augmentasi Gambar",
            "detail": "Kiri → Kanan" if eye_side == "left" else "Tidak di-flip (Sisi Kanan)",
            "value": "Horizontal Flip (flip_left_to_right)" if eye_side == "left" else "Bypass",
        },
        {
            "name": "Lokalisasi & Crop Konjungtiva",
            "detail": "Deteksi area kemerahan",
            "value": "BBox cropping (padding 15%)",
        },
        {
            "name": "Resize & Padding",
            "detail": "224×224 px dengan padding",
            "value": "Aspek rasio dipertahankan + pad border hitam",
        },
        {
            "name": "Test-Time Augmentation (TTA)",
            "detail": "8x forward passes",
            "value": "Rotasi, flip, translasi, dan brightness random",
        },
        {
            "name": "EfficientNetB0 end-to-end",
            "detail": "ImageNet backbone + GeM Pooling",
            "value": "Ekstraksi fitur langsung ke classifier head",
        },
        {
            "name": "ONNX Runtime Inference",
            "detail": f"Threshold: {result['threshold']:.2f}",
            "value": f"Rata-rata P(anemia)={result['probabilitas']:.4f}",
        },
    ]

    preprocessing = {
        "steps": steps_list,
        "pipeline_summary": {
            "backbone": "EfficientNetB0 (ONNX)",
            "img_feat_dim": 1280,
            "pca_components": 0,
            "total_features": 1280,
            "classifier": "Dense Head (Sigmoid)",
            "threshold": result["threshold"],
        },
    }

    full_result = {
        "id": scan_id,
        "timestamp": timestamp,
        "image_url": f"/uploads/{image_id}",
        "cropped_image_url": cropped_image_url,
        "symptoms": symptoms_list,
        "user_email": user_email if user_email else "",
        "preprocessing": preprocessing,
        "eye_side": eye_side,
        **result,
    }

    # Simpan ke PostgreSQL
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                INSERT INTO public.scans (
                    scan_id, user_email, nama, umur, gender, age_group, symptoms,
                    prediksi, probabilitas, threshold, risk_level, keterangan,
                    image_url, timestamp, img_feat_dim, pca_components, age_enc,
                    gender_enc, age_map_val, eye_side, cropped_image_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    scan_id,
                    user_email if user_email else "",
                    name,
                    age,
                    result["gender"],
                    result["age_group"],
                    symptoms,
                    result["prediksi"],
                    float(result["probabilitas"]),
                    float(result["threshold"]),
                    result["risk_level"],
                    result["keterangan"],
                    f"/uploads/{image_id}",
                    timestamp,
                    1280,
                    0,
                    0,
                    0,
                    0,
                    eye_side,
                    cropped_image_url,
                ),
            )
            conn.commit()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Gagal menyimpan hasil scan ke PostgreSQL: {str(e)}"
        )

    return full_result


# ─── History Endpoint ─────────────────────────────────────────────
@app.get("/api/history")
async def get_history(
    role: str = Query("user"),
    email: str = Query(""),
):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if role == "admin":
                cursor.execute("SELECT * FROM public.scans ORDER BY created_at DESC")
            elif email:
                cursor.execute(
                    "SELECT * FROM public.scans WHERE user_email = %s ORDER BY created_at DESC",
                    (email,),
                )
            else:
                return []
            rows = cursor.fetchall()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    result = []
    for row in rows:
        symptoms_list = [s.strip() for s in row.get("symptoms", "").split(",") if s.strip()]
        result.append(
            {
                "id": row["scan_id"],
                "timestamp": row["timestamp"],
                "image_url": row.get("image_url", ""),
                "symptoms": symptoms_list,
                "user_email": row.get("user_email", ""),
                "nama": row["nama"],
                "umur": row["umur"],
                "gender": row["gender"],
                "age_group": row["age_group"],
                "prediksi": row["prediksi"],
                "probabilitas": row["probabilitas"],
                "threshold": row["threshold"],
                "risk_level": row["risk_level"],
                "keterangan": row["keterangan"],
                "preprocessing": {
                    "pipeline_summary": {
                        "backbone": "EfficientNetB0 (ONNX)",
                        "img_feat_dim": row.get("img_feat_dim", 1280),
                        "pca_components": row.get("pca_components", 0),
                        "total_features": 1280,
                        "classifier": "Dense Head (Sigmoid)",
                        "threshold": row["threshold"],
                    }
                },
            }
        )

    return result


# ─── Stats Endpoint ───────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT prediksi FROM public.scans")
            scans = cursor.fetchall()
            cursor.execute("SELECT id FROM public.users")
            users = cursor.fetchall()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    total = len(scans)
    anemia = sum(1 for s in scans if s["prediksi"] == "Anemia")
    return {
        "total_scans": total,
        "anemia_count": anemia,
        "normal_count": total - anemia,
        "total_users": len(users),
    }


# ─── Static Files & Routes ────────────────────────────────────────
@app.get("/")
async def read_index():
    return FileResponse("index.html")


app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/logo.png")
async def get_logo():
    return FileResponse("assets/logo1.png")


@app.get("/logo1.png")
async def get_logo_exact():
    return FileResponse("assets/logo1.png")


@app.get("/Backgorund.png")
async def get_background_legacy():
    """Route lama — redirect ke nama file yang sudah diperbaiki."""
    return FileResponse("assets/background.png")


@app.get("/background.png")
async def get_background():
    return FileResponse("assets/background.png")


@app.get("/loading.mp4")
async def get_loading_video():
    return FileResponse("assets/loading.mp4", media_type="video/mp4")


@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("assets/logo2.png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

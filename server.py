
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import uuid
import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import bcrypt
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor

# Import logic from inference.py
from inference import load_model_components, predict_from_pil
from augmentation import flip_left_to_right
import numpy as np
import cv2

# ─── PostgreSQL Database Connection ──────────────────────────────
def get_db_connection():
    # 1. Coba baca dari file secrets.toml Streamlit jika ada
    try:
        import tomllib
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "rb") as f:
                sec = tomllib.load(f)
                if "postgres" in sec:
                    p = sec["postgres"]
                    if "url" in p:
                        return psycopg2.connect(p["url"])
                    return psycopg2.connect(
                        host=p.get("host", "127.0.0.1"),
                        port=p.get("port", "5432"),
                        database=p.get("database", "hemoscan"),
                        user=p.get("user", "postgres"),
                        password=p.get("password", "Sweethome123")
                    )
    except Exception:
        pass

    # 2. Coba dari environment variable POSTGRES_URL
    postgres_url = os.environ.get("POSTGRES_URL")
    if postgres_url:
        print(f"[DB Connection] Attempting PostgreSQL connection via URL.")
        return psycopg2.connect(postgres_url)

    # 3. Fallback ke parameter individual dengan default password 'Sweethome123'
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "hemoscan")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pwd = os.environ.get("POSTGRES_PASSWORD", "Sweethome123")
    
    print(f"[DB Connection] Attempting PostgreSQL connection: host={host}, port={port}, database={db}, user={user}, password={'*' * len(pwd)}")

    return psycopg2.connect(
        host=host,
        port=port,
        database=db,
        user=user,
        password=pwd
    )

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

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model components once
try:
    COMPONENTS = load_model_components()
    print("[Model] Components loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    COMPONENTS = None

# Uploads directory
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# ─── Helper: get preprocessing info from components ──────────────
def get_preprocessing_info():
    """Extract pipeline metadata for display in UI."""
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

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT id FROM public.users WHERE email = %s", (req.email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            raise HTTPException(status_code=409, detail="Email sudah terdaftar")
        
        password_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cursor.execute(
            """
            INSERT INTO public.users (email, password_hash, display_name, role)
            VALUES (%s, %s, %s, 'user')
            """,
            (req.email, password_hash, req.display_name.strip())
        )
        conn.commit()
        cursor.close()
        conn.close()
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "status": "success",
        "message": "Pendaftaran berhasil! Silakan masuk.",
        "email": req.email,
        "display_name": req.display_name.strip()
    }

# ─── Login Endpoint ───────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM public.users WHERE email = %s", (req.email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        import traceback
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
            "display_name": user["display_name"]
        }
    raise HTTPException(status_code=401, detail="Email atau password salah")

# ─── Link Scans Endpoint ──────────────────────────────────────────
@app.post("/api/link-scans")
async def link_scans(req: LinkScansRequest):
    if not req.email or not req.scan_ids:
        return {"status": "success", "message": "No scans to link"}

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        for scan_id in req.scan_ids:
            cursor.execute(
                """
                UPDATE public.scans
                SET user_email = %s
                WHERE scan_id = %s AND (user_email = '' OR user_email IS NULL)
                """,
                (req.email, scan_id)
            )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error while linking scans: {str(e)}")

    return {"status": "success", "message": f"Linked {len(req.scan_ids)} scans to {req.email}"}

# ─── Preprocessing Info Endpoint ─────────────────────────────────
@app.get("/api/preprocessing-info")
async def preprocessing_info():
    """Return the full ML pipeline configuration for UI display."""
    return get_preprocessing_info()


# ─── Image Validation Helper ──────────────────────────────────────
def validate_conjunctiva_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Validate whether the uploaded image looks like a conjunctiva (eye) photo.
    Uses color histogram heuristics to detect pinkish/reddish tissue.

    Returns (is_valid, reason).
    """
    import numpy as np

    img = pil_image.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # --- Check 1: Image should not be too dark or too bright overall ---
    mean_brightness = arr.mean()
    if mean_brightness < 30:
        return False, "Gambar terlalu gelap. Pastikan pencahayaan cukup saat mengambil foto konjungtiva."
    if mean_brightness > 245:
        return False, "Gambar terlalu terang/putih. Pastikan kamera fokus pada area konjungtiva."

    # --- Check 2: Red channel should generally dominate for skin/conjunctiva ---
    # For conjunctiva images, the red channel is typically higher than blue
    red_dominant_pixels = np.sum((r > b + 5) & (r > 60)) / (224 * 224)
    if red_dominant_pixels < 0.15:
        return False, "Foto yang dikirim bukan merupakan gambar konjungtiva mata. Silakan ambil ulang foto bagian dalam kelopak mata bawah."

    # --- Check 3: There should be some pinkish/reddish tissue pixels ---
    # Conjunctiva has pinkish-red color: R > G > B typically, with R being prominent
    pinkish_mask = (r > 80) & (r > g) & (g > b * 0.5) & (r - b > 15)
    pinkish_ratio = np.sum(pinkish_mask) / (224 * 224)
    if pinkish_ratio < 0.08:
        return False, "Foto yang dikirim bukan merupakan gambar konjungtiva mata. Silakan ambil ulang foto bagian dalam kelopak mata bawah."

    # --- Check 4: Image shouldn't be too uniform (solid color / blank) ---
    std_r, std_g, std_b = r.std(), g.std(), b.std()
    avg_std = (std_r + std_g + std_b) / 3
    if avg_std < 8:
        return False, "Gambar terdeteksi sebagai warna solid. Silakan ambil foto konjungtiva yang jelas."

    # --- Check 5: Not pure grayscale (e.g., text documents, B&W photos) ---
    channel_diff = np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()
    if channel_diff < 5:
        return False, "Gambar terdeteksi sebagai hitam-putih. Silakan kirim foto berwarna dari konjungtiva mata."

    return True, "OK"


def crop_conjunctiva(pil_image: Image.Image) -> Image.Image:
    """
    Locates and crops the conjunctiva palpebra inferior (pinkish/reddish area)
    using hierarchical color thresholding, vertical eye-region filtering, and contour union.
    """
    orig_w, orig_h = pil_image.size
    process_img = pil_image.convert('RGB').resize((224, 224))
    arr = np.array(process_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Define redness masks hierarchically from strictest to loosest
    masks = [
        # Very Strict Redness
        ("Very Strict RGB", (r > 120) & (r > g + 30) & (r - b > 50)),
        # Strict Redness
        ("Strict RGB", (r > 105) & (r > g + 20) & (r - b > 30)),
        # Medium Redness
        ("Medium RGB", (r > 100) & (r > g + 15) & (r - b > 25)),
        # Original Loose Redness
        ("Original Loose RGB", (r > 80) & (r > g) & (g > b * 0.5) & (r - b > 15))
    ]
    
    for mask_name, mask_expr in masks:
        mask_uint8 = (mask_expr.astype(np.uint8)) * 255
        
        # Apply a small morphological close to smooth contours and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # Filter contours by vertical position (keeping only those in the eye region: 30 < cy < 135)
        # to filter out hands/fingers pulling the eyelid (cy >= 135) and hair/headbands (cy <= 30)
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= 3:
                x, y, w, h = cv2.boundingRect(c)
                cy = y + h / 2.0
                if 30 < cy < 135:
                    valid_contours.append(c)
                    
        if valid_contours:
            # Combine all valid eye contours to get the union bounding box
            all_pts = np.vstack(valid_contours)
            ux, uy, uw, uh = cv2.boundingRect(all_pts)
            
            # Scale bounding box back to original image size and add 15% padding
            ymin_scaled = int(max(0, (uy - uh * 0.15) / 224 * orig_h))
            ymax_scaled = int(min(orig_h, (uy + uh + uh * 0.15) / 224 * orig_h))
            xmin_scaled = int(max(0, (ux - uw * 0.15) / 224 * orig_w))
            xmax_scaled = int(min(orig_w, (ux + uw + uw * 0.15) / 224 * orig_w))
            
            print(f"[Crop] Matched mask: {mask_name} (Contours={len(valid_contours)})")
            print(f"[Crop] Bounding box: {xmin_scaled}:{xmax_scaled}, {ymin_scaled}:{ymax_scaled}")
            return pil_image.crop((xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled))
                
    # Fallback to center crop if no suitable region is found
    print("[Crop] No suitable eye contours found. Using center crop fallback.")
    crop_size = min(orig_w, orig_h)
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    return pil_image.crop((left, top, left + crop_size, top + crop_size))



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

    try:
        # Read image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # ── Validate image is a conjunctiva photo ──────────────
        is_valid, validation_msg = validate_conjunctiva_image(pil_image)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_msg)

        # Apply horizontal flip (augmentation) only if eye_side is left
        if eye_side == "left":
            img_np = np.array(pil_image)
            flipped_np = flip_left_to_right(img_np)
            pil_image = Image.fromarray(flipped_np)

        # Save image for history
        image_id = f"{uuid.uuid4()}.png"
        image_path = os.path.join(UPLOADS_DIR, image_id)
        pil_image.save(image_path)

        # Crop conjunctiva palpebra inferior
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

        # Run prediction on cropped image
        result = predict_from_pil(cropped_image, name, age, gender, COMPONENTS)

        # Build metadata
        scan_id = f"#SCN-{uuid.uuid4().hex[:4].upper()}"
        timestamp = datetime.datetime.now().strftime("%b %d, %Y • %H:%M")
        symptoms_list = symptoms.split(",") if symptoms else []

        n_pca = 0
        age_enc = 0
        gender_enc = 0

        # Preprocessing pipeline details to store and return
        steps_list = [
            {
                "name": "Augmentasi Gambar",
                "detail": "Kiri → Kanan" if eye_side == "left" else "Tidak di-flip (Sisi Kanan)",
                "value": "Horizontal Flip (flip_left_to_right)" if eye_side == "left" else "Bypass"
            },
            {
                "name": "Lokalisasi & Crop Konjungtiva",
                "detail": "Deteksi area kemerahan",
                "value": "BBox cropping (padding 15%)"
            },
            {
                "name": "Resize & Padding",
                "detail": "224×224 px dengan padding",
                "value": "Aspek rasio dipertahankan + pad border hitam"
            },
            {
                "name": "Test-Time Augmentation (TTA)",
                "detail": "8x forward passes",
                "value": "Rotasi, flip, translasi, dan brightness random"
            },
            {
                "name": "EfficientNetB0 end-to-end",
                "detail": "ImageNet backbone + GeM Pooling",
                "value": "Ekstraksi fitur langsung ke classifier head"
            },
            {
                "name": "ONNX Runtime Inference",
                "detail": f"Threshold: {result['threshold']:.2f}",
                "value": f"Rata-rata P(anemia)={result['probabilitas']:.4f}"
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
            }
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
            **result
        }

        # ── Save to PostgreSQL ─────────────────────────────────────
        try:
            conn = get_db_connection()
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
                    n_pca,
                    int(age_enc),
                    int(gender_enc),
                    int(age_enc),
                    eye_side,
                    cropped_image_url
                )
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Gagal menyimpan hasil scan ke PostgreSQL: {str(e)}")

        return full_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── History Endpoint (from Supabase, role-aware) ────────────────
@app.get("/api/history")
async def get_history(
    role: str = Query("user"),
    email: str = Query("")
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        if role == "admin":
            cursor.execute("SELECT * FROM public.scans ORDER BY created_at DESC")
        elif email:
            cursor.execute("SELECT * FROM public.scans WHERE user_email = %s ORDER BY created_at DESC", (email,))
        else:
            cursor.close()
            conn.close()
            return []
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    result = []
    for row in rows:
        symptoms_list = [s.strip() for s in row.get("symptoms", "").split(",") if s.strip()]
        result.append({
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
                    "backbone": "EfficientNetB0 (ImageNet)",
                    "img_feat_dim": row.get("img_feat_dim", 1280),
                    "pca_components": row.get("pca_components", 0),
                    "total_features": 3 + row.get("pca_components", 0),
                    "classifier": "XGBoost",
                    "threshold": row["threshold"],
                }
            }
        })

    return result

# ─── Stats Endpoint (admin summary) ──────────────────────────────
@app.get("/api/stats")
async def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT prediksi FROM public.scans")
        scans = cursor.fetchall()
        cursor.execute("SELECT id FROM public.users")
        users = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    total = len(scans)
    anemia = sum(1 for s in scans if s["prediksi"] == "Anemia")
    return {
        "total_scans": total,
        "anemia_count": anemia,
        "normal_count": total - anemia,
        "total_users": len(users)
    }

# ─── Static Files & Routes ────────────────────────────────────────
@app.get("/")
async def read_index():
    return FileResponse("index.html")

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/logo.png")
async def get_logo():
    return FileResponse("assets/logo1.png")

@app.get("/logo1.png")
async def get_logo_exact():
    return FileResponse("assets/logo1.png")

@app.get("/Backgorund.png")
async def get_background():
    return FileResponse("assets/Backgorund.png")

@app.get("/loading.mp4")
async def get_loading_video():
    return FileResponse("assets/loading.mp4", media_type="video/mp4")

@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("assets/logo2.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

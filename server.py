
import os
import json
import uuid
import sqlite3
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

# Import logic from inference.py
from inference import load_model_components, predict_from_pil

# ─── Database Setup ──────────────────────────────────────────────
DB_FILE = "hemoscan.db"

def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with users and scans tables."""
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Scans table — stores all prediction results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT UNIQUE NOT NULL,
            user_email TEXT NOT NULL DEFAULT '',
            nama TEXT NOT NULL,
            umur INTEGER NOT NULL,
            gender TEXT NOT NULL,
            age_group TEXT NOT NULL,
            symptoms TEXT NOT NULL DEFAULT '',
            prediksi TEXT NOT NULL,
            probabilitas REAL NOT NULL,
            threshold REAL NOT NULL,
            risk_level TEXT NOT NULL,
            keterangan TEXT NOT NULL,
            image_url TEXT NOT NULL DEFAULT '',
            timestamp TEXT NOT NULL,
            -- preprocessing pipeline info
            img_feat_dim INTEGER NOT NULL DEFAULT 1280,
            pca_components INTEGER NOT NULL DEFAULT 0,
            age_enc INTEGER NOT NULL DEFAULT 0,
            gender_enc INTEGER NOT NULL DEFAULT 0,
            age_map_val INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

    # Seed default admin account if no users exist
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    count = cursor.fetchone()["cnt"]
    if count == 0:
        admin_hash = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cursor.execute(
            "INSERT INTO users (email, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            ("admin@hemoscan.com", admin_hash, "Administrator", "admin")
        )
        conn.commit()
        print("[DB] Default admin account created: admin@hemoscan.com / admin123")

    conn.close()

# Initialize database on startup
init_db()

# ─── Pydantic Models ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str

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
    pca = COMPONENTS.get("pca", None)
    n_pca = int(pca.n_components_) if pca is not None and hasattr(pca, "n_components_") else 0
    age_map = config.get("age_map", {})
    gender_map = config.get("gender_map", {})
    best_thresh = config.get("best_thresh", 0.5)
    return {
        "img_input_size": [224, 224, 3],
        "backbone": "EfficientNetB0",
        "img_feat_dim": 1280,
        "pca_components": n_pca,
        "tabular_features": ["umur", "gender_enc", "age_group_enc"],
        "age_map": age_map,
        "gender_map": gender_map,
        "classifier": "XGBoost",
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

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (req.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Email sudah terdaftar")

    password_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    cursor.execute(
        "INSERT INTO users (email, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
        (req.email, password_hash, req.display_name.strip(), "user")
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": "Pendaftaran berhasil! Silakan masuk.",
        "email": req.email,
        "display_name": req.display_name.strip()
    }

# ─── Login Endpoint ───────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (req.email,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(req.password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return {
            "status": "success",
            "message": "Login successful",
            "email": user["email"],
            "role": user["role"],
            "display_name": user["display_name"]
        }
    raise HTTPException(status_code=401, detail="Email atau password salah")

# ─── Preprocessing Info Endpoint ─────────────────────────────────
@app.get("/api/preprocessing-info")
async def preprocessing_info():
    """Return the full ML pipeline configuration for UI display."""
    return get_preprocessing_info()

# ─── Predict Endpoint ─────────────────────────────────────────────
@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    symptoms: str = Form(""),
    user_email: str = Form("")
):
    if COMPONENTS is None:
        raise HTTPException(status_code=500, detail="Model components not loaded")

    try:
        # Read image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Save image for history
        image_id = f"{uuid.uuid4()}.png"
        image_path = os.path.join(UPLOADS_DIR, image_id)
        pil_image.save(image_path)

        # Run prediction
        result = predict_from_pil(pil_image, name, age, gender, COMPONENTS)

        # Build metadata
        scan_id = f"#SCN-{uuid.uuid4().hex[:4].upper()}"
        timestamp = datetime.datetime.now().strftime("%b %d, %Y • %H:%M")
        symptoms_list = symptoms.split(",") if symptoms else []

        # Get PCA info
        pca = COMPONENTS.get("pca", None)
        n_pca = int(pca.n_components_) if pca is not None and hasattr(pca, "n_components_") else 0
        config = COMPONENTS.get("config", {})
        age_map = config.get("age_map", {})
        gender_map = config.get("gender_map", {})

        from inference import hitung_age_group
        age_group = hitung_age_group(age)
        age_enc = age_map.get(age_group, 1)
        gender_enc = gender_map.get(gender, 0)

        # Preprocessing pipeline details to store and return
        preprocessing = {
            "steps": [
                {
                    "name": "Resize & Konversi RGB",
                    "detail": "224×224 px",
                    "value": "PIL → RGB → resize(224,224)"
                },
                {
                    "name": "EfficientNetB0 Feature Extraction",
                    "detail": f"{1280} dimensi",
                    "value": "ImageNet weights · Global Avg Pool"
                },
                {
                    "name": "PCA Dimensionality Reduction",
                    "detail": f"1280 → {n_pca} komponen",
                    "value": f"{n_pca} principal components"
                },
                {
                    "name": "Tabular Encoding",
                    "detail": "Umur + Gender + Kelompok Usia",
                    "value": f"umur={age}, gender_enc={gender_enc}, age_enc={age_enc}"
                },
                {
                    "name": "Feature Concatenation",
                    "detail": f"3 tabular + {n_pca} PCA = {3+n_pca} total fitur",
                    "value": f"X_input shape: (1, {3+n_pca})"
                },
                {
                    "name": "XGBoost Classifier",
                    "detail": f"Threshold: {result['threshold']:.2f}",
                    "value": f"P(anemia)={result['probabilitas']:.4f}"
                },
            ],
            "pipeline_summary": {
                "backbone": "EfficientNetB0 (ImageNet)",
                "img_feat_dim": 1280,
                "pca_components": n_pca,
                "total_features": 3 + n_pca,
                "classifier": "XGBoost",
                "threshold": result["threshold"],
            }
        }

        full_result = {
            "id": scan_id,
            "timestamp": timestamp,
            "image_url": f"/uploads/{image_id}",
            "symptoms": symptoms_list,
            "user_email": user_email if user_email else "",
            "preprocessing": preprocessing,
            **result
        }

        # ── Save to SQLite DB ──────────────────────────────────────
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scans (
                scan_id, user_email, nama, umur, gender, age_group,
                symptoms, prediksi, probabilitas, threshold, risk_level,
                keterangan, image_url, timestamp,
                img_feat_dim, pca_components, age_enc, gender_enc, age_map_val
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_id,
            user_email if user_email else "",
            name, age,
            result["gender"],
            result["age_group"],
            symptoms,
            result["prediksi"],
            result["probabilitas"],
            result["threshold"],
            result["risk_level"],
            result["keterangan"],
            f"/uploads/{image_id}",
            timestamp,
            1280,
            n_pca,
            age_enc,
            gender_enc,
            age_enc,
        ))
        conn.commit()
        conn.close()

        return full_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── History Endpoint (from DB, role-aware) ──────────────────────
@app.get("/api/history")
async def get_history(
    role: str = Query("user"),
    email: str = Query("")
):
    conn = get_db()
    cursor = conn.cursor()

    if role == "admin":
        cursor.execute("""
            SELECT * FROM scans ORDER BY created_at DESC
        """)
    elif email:
        cursor.execute("""
            SELECT * FROM scans WHERE user_email = ? ORDER BY created_at DESC
        """, (email,))
    else:
        conn.close()
        return []

    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        symptoms_list = [s.strip() for s in row["symptoms"].split(",") if s.strip()]
        result.append({
            "id": row["scan_id"],
            "timestamp": row["timestamp"],
            "image_url": row["image_url"],
            "symptoms": symptoms_list,
            "user_email": row["user_email"],
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
                    "img_feat_dim": row["img_feat_dim"],
                    "pca_components": row["pca_components"],
                    "total_features": 3 + row["pca_components"],
                    "classifier": "XGBoost",
                    "threshold": row["threshold"],
                }
            }
        })

    return result

# ─── Stats Endpoint (admin summary) ──────────────────────────────
@app.get("/api/stats")
async def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM scans")
    total = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as anemia FROM scans WHERE prediksi='Anemia'")
    anemia = cursor.fetchone()["anemia"]
    cursor.execute("SELECT COUNT(*) as total_users FROM users")
    total_users = cursor.fetchone()["total_users"]
    conn.close()
    return {
        "total_scans": total,
        "anemia_count": anemia,
        "normal_count": total - anemia,
        "total_users": total_users
    }

# ─── Static Files & Routes ────────────────────────────────────────
@app.get("/")
async def read_index():
    return FileResponse("index.html")

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/logo.png")
async def get_logo():
    return FileResponse("logo 1.png")

@app.get("/logo 1.png")
async def get_logo_exact():
    return FileResponse("logo 1.png")

@app.get("/Backgorund.png")
async def get_background():
    return FileResponse("Backgorund.png")

@app.get("/loading.mp4")
async def get_loading_video():
    return FileResponse("loading.mp4", media_type="video/mp4")

@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("logo 2.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

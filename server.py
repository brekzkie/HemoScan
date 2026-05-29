
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
    """Initialize the database with the users table."""
    conn = get_db()
    cursor = conn.cursor()
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
except Exception as e:
    print(f"Error loading models: {e}")
    COMPONENTS = None

# History storage (simple JSON for now)
HISTORY_FILE = "scan_history.json"
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def save_to_history(data: dict):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.insert(0, data)  # Add to top
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ─── Register Endpoint ───────────────────────────────────────────
@app.post("/api/register")
async def register(req: RegisterRequest):
    # Validate input
    if not req.email or not req.password or not req.display_name:
        raise HTTPException(status_code=400, detail="Semua kolom wajib diisi")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password minimal 6 karakter")

    if len(req.display_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Nama lengkap minimal 2 karakter")

    conn = get_db()
    cursor = conn.cursor()

    # Check if email already exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (req.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Email sudah terdaftar")

    # Hash password with bcrypt
    password_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    # Insert new user
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

# ─── Login Endpoint (database-backed with bcrypt) ────────────────
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

# ─── Predict Endpoint (now tracks user_email) ────────────────────
@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),        # 'M' or 'F'
    symptoms: str = Form(""),       # Comma separated
    user_email: str = Form("")      # Email of logged-in user (empty if guest)
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
        
        # Add metadata
        scan_id = f"#SCN-{uuid.uuid4().hex[:4].upper()}"
        timestamp = datetime.datetime.now().strftime("%b %d, %Y • %H:%M")
        
        full_result = {
            "id": scan_id,
            "timestamp": timestamp,
            "image_url": f"/uploads/{image_id}",
            "symptoms": symptoms.split(",") if symptoms else [],
            "user_email": user_email if user_email else "",
            **result
        }
        
        # Save to history
        save_to_history(full_result)
        
        return full_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── History Endpoint (filtered by role) ──────────────────────────
@app.get("/api/history")
async def get_history(
    role: str = Query("user"),
    email: str = Query("")
):
    if not os.path.exists(HISTORY_FILE):
        return []
    
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    # Admin sees everything
    if role == "admin":
        return history
    
    # Regular user sees only their own scans (matched by user_email)
    if email:
        filtered = [h for h in history if h.get("user_email", "") == email]
        return filtered
    
    return []

# Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# Serve uploaded files
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


import os
import json
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

# Import logic from inference.py
from inference import load_model_components, predict_from_pil

# ─── Account System ──────────────────────────────────────────────
# Predefined accounts: admin and user roles
ACCOUNTS = {
    "admin@hemoscan.com": {
        "password": "admin123",
        "role": "admin",
        "display_name": "Administrator"
    },
    "user@hemoscan.com": {
        "password": "user123",
        "role": "user",
        "display_name": "User Demo"
    },
    "dokter@hemoscan.com": {
        "password": "dokter123",
        "role": "user",
        "display_name": "Dr. Amelia"
    },
}

class LoginRequest(BaseModel):
    email: str
    password: str

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

# ─── Login Endpoint (multi-account with roles) ───────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    account = ACCOUNTS.get(req.email)
    if account and account["password"] == req.password:
        return {
            "status": "success",
            "message": "Login successful",
            "email": req.email,
            "role": account["role"],
            "display_name": account["display_name"]
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

@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("logo 2.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

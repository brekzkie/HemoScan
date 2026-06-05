"""
database.py — PostgreSQL connector untuk HemoScan
Menggantikan fungsi SQLite di server.py
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

import bcrypt
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor


# ─── PostgreSQL Database Connection ──────────────────────────────
def get_db_connection():
    """Inisialisasi koneksi PostgreSQL."""
    # Check streamlit secrets
    if "postgres" in st.secrets:
        sec = st.secrets["postgres"]
        if "url" in sec:
            return psycopg2.connect(sec["url"])
        return psycopg2.connect(
            host=sec.get("host", "127.0.0.1"),
            port=sec.get("port", "5432"),
            database=sec.get("database", "hemoscan"),
            user=sec.get("user", "postgres"),
            password=sec.get("password", "Sweethome123")
        )
    # Fallback to env vars
    postgres_url = os.environ.get("POSTGRES_URL")
    if postgres_url:
        return psycopg2.connect(postgres_url)
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        database=os.environ.get("POSTGRES_DB", "hemoscan"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "Sweethome123")
    )


# ─── User Operations ─────────────────────────────────────────────
def register_user(email: str, password: str, display_name: str) -> dict:
    """
    Daftarkan user baru ke tabel users.
    Returns dict: {"success": bool, "message": str}
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Cek apakah email sudah terdaftar
        cursor.execute("SELECT id FROM public.users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return {"success": False, "message": "Email sudah terdaftar"}

        # Hash password
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Insert user baru
        cursor.execute(
            """
            INSERT INTO public.users (email, password_hash, display_name, role)
            VALUES (%s, %s, %s, 'user')
            """,
            (email, password_hash, display_name.strip())
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {"success": True, "message": "Pendaftaran berhasil!"}
    except Exception as e:
        return {"success": False, "message": f"Gagal mendaftar: {str(e)}"}


def login_user(email: str, password: str) -> dict:
    """
    Verifikasi login user.
    Returns dict: {"success": bool, "user": dict | None, "message": str}
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM public.users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        return {"success": False, "user": None, "message": f"Database error: {str(e)}"}

    if not user:
        return {"success": False, "user": None, "message": "Email atau password salah"}

    if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return {
            "success": True,
            "user": {
                "email": user["email"],
                "display_name": user["display_name"],
                "role": user["role"]
            },
            "message": "Login berhasil"
        }
    return {"success": False, "user": None, "message": "Email atau password salah"}


# ─── Scan Operations ─────────────────────────────────────────────
def save_scan(scan_data: dict) -> dict:
    """
    Simpan hasil scan ke tabel scans.
    scan_data harus mengandung semua kolom yang diperlukan.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            INSERT INTO public.scans (
                scan_id, user_email, nama, umur, gender, age_group, symptoms,
                prediksi, probabilitas, threshold, risk_level, keterangan,
                image_url, timestamp, img_feat_dim, pca_components, age_enc,
                gender_enc, age_map_val
            ) VALUES (
                %(scan_id)s, %(user_email)s, %(nama)s, %(umur)s, %(gender)s, %(age_group)s, %(symptoms)s,
                %(prediksi)s, %(probabilitas)s, %(threshold)s, %(risk_level)s, %(keterangan)s,
                %(image_url)s, %(timestamp)s, %(img_feat_dim)s, %(pca_components)s, %(age_enc)s,
                %(gender_enc)s, %(age_map_val)s
            ) RETURNING *
            """,
            scan_data
        )
        row = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        if row:
            return {"success": True, "data": dict(row)}
        return {"success": False, "data": None}
    except Exception as e:
        return {"success": False, "data": None}


def get_history(role: str, email: str = "") -> list:
    """
    Ambil riwayat scan berdasarkan role.
    - admin: semua scan
    - user: hanya scan milik email tersebut
    """
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
        return []

    history = []
    for row in rows:
        symptoms_list = [s.strip() for s in row.get("symptoms", "").split(",") if s.strip()]
        history.append({
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
            "pca_components": row.get("pca_components", 0),
            "img_feat_dim": row.get("img_feat_dim", 1280),
        })
    return history


def get_stats() -> dict:
    """Ambil statistik untuk dashboard admin."""
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
        return {
            "total_scans": 0,
            "anemia_count": 0,
            "normal_count": 0,
            "total_users": 0
        }

    total = len(scans)
    anemia = sum(1 for s in scans if s["prediksi"] == "Anemia")

    return {
        "total_scans": total,
        "anemia_count": anemia,
        "normal_count": total - anemia,
        "total_users": len(users)
    }


def get_all_users() -> list:
    """Ambil semua user (untuk admin)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT email, display_name, role, created_at FROM public.users ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return []


def get_scan_trend() -> list:
    """Ambil data scan dengan timestamp untuk grafik tren."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT prediksi, created_at, gender, age_group, probabilitas FROM public.scans ORDER BY created_at")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return []

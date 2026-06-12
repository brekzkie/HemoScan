# ================================================================
#  config.py
#  Konfigurasi terpusat untuk HemoScan
# ================================================================

import os
from dotenv import load_dotenv

load_dotenv(override=True)


# ─── Database (PostgreSQL) ───────────────────────────────────────
def _get_postgres_url_from_secrets() -> str | None:
    """Coba baca URL PostgreSQL dari .streamlit/secrets.toml."""
    try:
        import tomllib
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "rb") as f:
                sec = tomllib.load(f)
                if "postgres" in sec:
                    p = sec["postgres"]
                    if "url" in p:
                        return p["url"]
    except Exception:
        pass
    return None


# Konfigurasi DB — tidak ada default password (wajib set via env)
POSTGRES_URL: str | None = (
    _get_postgres_url_from_secrets()
    or os.environ.get("POSTGRES_URL")
)
POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "hemoscan")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")


# ─── Default Admin Configuration ──────────────────────────────────
ADMIN_EMAIL: str = os.environ.get("ADMIN_EMAIL", "admin@hemoscan.com")
ADMIN_PASSWORD: str = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_DISPLAY_NAME: str = os.environ.get("ADMIN_DISPLAY_NAME", "Administrator")


# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(BASE_DIR, "model_output")
UPLOADS_DIR: str = os.path.join(BASE_DIR, "uploads")
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")


# ─── Model ───────────────────────────────────────────────────────
ONNX_MODEL_FILE: str = "anemia_model_final.onnx"
DEFAULT_THRESHOLD: float = 0.50
DEFAULT_N_TTA: int = 8


# ─── Database Connection ─────────────────────────────────────────
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager


@contextmanager
def get_db_connection():
    """
    Context manager untuk koneksi PostgreSQL.
    Otomatis menutup koneksi saat selesai atau terjadi error.

    Penggunaan:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(...)
            conn.commit()
    """
    conn = None
    try:
        if POSTGRES_URL:
            conn = psycopg2.connect(POSTGRES_URL)
        else:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
            )
        yield conn
    finally:
        if conn is not None:
            conn.close()

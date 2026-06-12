"""
init_db.py — Inisialisasi database PostgreSQL untuk HemoScan
Membuat database jika belum ada, lalu menjalankan schema SQL.
"""

import os
import sys

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load .env hanya jika ada (di Docker pakai env vars langsung dari compose)
load_dotenv(override=False)


def init_postgres() -> bool:
    """
    Inisialisasi PostgreSQL: buat database jika belum ada,
    lalu jalankan postgres_schema.sql untuk membuat tabel dan seed admin.

    Returns:
        True jika berhasil, False jika gagal.
    """
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "hemoscan")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pwd = os.environ.get("POSTGRES_PASSWORD", "")

    if not pwd:
        print("WARNING: POSTGRES_PASSWORD tidak di-set. Pastikan env var sudah dikonfigurasi.")

    print(f"Connecting to PostgreSQL server at {host}:{port} as user '{user}'...")

    # 1. Connect ke database 'postgres' default untuk cek/buat database target
    try:
        conn = psycopg2.connect(
            host=host, port=port, database="postgres", user=user, password=pwd
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to default 'postgres' database: {e}")
        print("Pastikan PostgreSQL berjalan dan kredensial benar.")
        return False

    # 2. Cek apakah database sudah ada
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
    exists = cursor.fetchone()

    if not exists:
        print(f"Database '{db_name}' belum ada. Membuat...")
        try:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' berhasil dibuat.")
        except Exception as e:
            print(f"Error membuat database '{db_name}': {e}")
            cursor.close()
            conn.close()
            return False
    else:
        print(f"Database '{db_name}' sudah ada.")

    cursor.close()
    conn.close()

    # 3. Connect ke database target dan jalankan schema
    print(f"Menjalankan schema di database '{db_name}'...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, database=db_name, user=user, password=pwd
        )
        cursor = conn.cursor()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(base_dir, "postgres_schema.sql")
        if not os.path.exists(schema_path):
            print(f"Error: Schema file '{schema_path}' tidak ditemukan.")
            return False

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        cursor.execute(schema_sql)
        conn.commit()

        # Seed admin secara dinamis dari environment variables
        admin_email = os.environ.get("ADMIN_EMAIL", "admin@hemoscan.com")
        admin_pwd = os.environ.get("ADMIN_PASSWORD", "admin123")
        admin_name = os.environ.get("ADMIN_DISPLAY_NAME", "Administrator")

        print(f"Seeding admin default ({admin_email})...")
        try:
            import bcrypt
            hashed_pwd = bcrypt.hashpw(admin_pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            
            cursor.execute(
                """
                INSERT INTO public.users (email, password_hash, display_name, role)
                VALUES (%s, %s, %s, 'admin')
                ON CONFLICT (email) 
                DO UPDATE SET password_hash = EXCLUDED.password_hash, display_name = EXCLUDED.display_name;
                """,
                (admin_email, hashed_pwd, admin_name)
            )
            conn.commit()
            print("Admin default berhasil di-seed (dan di-update jika ada perubahan kredensial).")
        except Exception as err_seed:
            print(f"Error seeding admin: {err_seed}")
            # Kita tidak ingin membatalkan seluruh init jika hanya seeding admin gagal (misal library bcrypt bermasalah),
            # tetapi karena bcrypt adalah prasyarat, ini krusial.

        print("Schema database berhasil diinisialisasi.")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error inisialisasi schema di database '{db_name}': {e}")
        return False


if __name__ == "__main__":
    success = init_postgres()
    if success:
        print("Database setup selesai!")
    else:
        print("Database setup gagal.")
        sys.exit(1)

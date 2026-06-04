import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load .env hanya jika ada (di Docker pakai env vars langsung dari compose)
load_dotenv(override=False)

def init_postgres():
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "hemoscan")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pwd = os.environ.get("POSTGRES_PASSWORD", "Sweethome123")

    print(f"Connecting to PostgreSQL server at {host}:{port} as user '{user}'...")

    # Connect ke default 'postgres' database untuk cek/buat database target
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=pwd
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to default 'postgres' database: {e}")
        print("Pastikan PostgreSQL berjalan dan kredensial benar.")
        return False

    # Cek apakah database sudah ada
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

    # Connect ke database target dan jalankan schema
    print(f"Menjalankan schema di database '{db_name}'...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=db_name,
            user=user,
            password=pwd
        )
        cursor = conn.cursor()

        # Cari postgres_schema.sql relatif terhadap file ini
        base_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(base_dir, "postgres_schema.sql")
        if not os.path.exists(schema_path):
            print(f"Error: Schema file '{schema_path}' tidak ditemukan.")
            return False

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        cursor.execute(schema_sql)
        conn.commit()
        print("Schema database berhasil diinisialisasi (tabel dibuat, admin di-seed).")

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

#!/bin/sh
# ================================================================
# entrypoint.sh — HemoScan startup script
# 1. Tunggu PostgreSQL siap
# 2. Inisialisasi database (buat tabel + seed admin)
# 3. Jalankan FastAPI server
# ================================================================

set -e

echo "================================================"
echo "   HemoScan — Starting Application"
echo "================================================"

# ── 1. Tunggu PostgreSQL siap ────────────────────────────────────
echo "[1/3] Waiting for PostgreSQL to be ready..."

until python -c "
import psycopg2, os, sys
try:
    conn = psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'db'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        database='postgres',
        user=os.environ.get('POSTGRES_USER', 'postgres'),
        password=os.environ.get('POSTGRES_PASSWORD', 'hemoscan_pass')
    )
    conn.close()
    print('PostgreSQL is ready!')
    sys.exit(0)
except Exception as e:
    print(f'Waiting... ({e})')
    sys.exit(1)
"; do
    sleep 2
done

# ── 2. Inisialisasi database ────────────────────────────────────
echo "[2/3] Initializing database schema..."
python database/init_db.py

# ── 3. Jalankan server ──────────────────────────────────────────
echo "[3/3] Starting FastAPI server on port 8000..."
echo "================================================"
echo "   App running at: http://localhost:8000"
echo "================================================"

exec python server.py

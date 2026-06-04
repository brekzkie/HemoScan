-- Hapus tabel lama jika sudah ada (agar schema ter-reset bersih)
DROP TABLE IF EXISTS public.scans CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;

-- Tabel Users
CREATE TABLE IF NOT EXISTS public.users (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabel Scans
CREATE TABLE IF NOT EXISTS public.scans (
    id              BIGSERIAL PRIMARY KEY,
    scan_id         TEXT UNIQUE NOT NULL,
    user_email      TEXT NOT NULL DEFAULT '',
    nama            TEXT NOT NULL,
    umur            INTEGER NOT NULL,
    gender          TEXT NOT NULL,
    age_group       TEXT NOT NULL,
    symptoms        TEXT NOT NULL DEFAULT '',
    prediksi        TEXT NOT NULL,
    probabilitas    FLOAT NOT NULL,
    threshold       FLOAT NOT NULL,
    risk_level      TEXT NOT NULL,
    keterangan      TEXT NOT NULL,
    image_url       TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL,
    img_feat_dim    INTEGER NOT NULL DEFAULT 1280,
    pca_components  INTEGER NOT NULL DEFAULT 0,
    age_enc         INTEGER NOT NULL DEFAULT 0,
    gender_enc      INTEGER NOT NULL DEFAULT 0,
    age_map_val     INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default admin (password: admin123)
INSERT INTO public.users (email, password_hash, display_name, role)
VALUES (
    'admin@hemoscan.com',
    '$2b$12$p26FcM4N/SKdIEOM7GZSc.blLGaSXOqmYjG3VycLXthwRrCNvNbny',
    'Administrator',
    'admin'
) ON CONFLICT (email) DO NOTHING;

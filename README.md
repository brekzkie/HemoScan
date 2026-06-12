# 🚀 HEMOSCAN — Anemia Detection System

<img width="3840" height="3840" alt="image" src="https://github.com/user-attachments/assets/650e0941-d8bb-4ad3-8754-f010a91327aa" />

HemoScan adalah solusi berbasis kecerdasan buatan (AI) untuk skrining anemia secara non-invasif melalui analisis citra konjungtiva mata menggunakan teknik *Deep Learning*.

---

## 🏗️ Arsitektur & Pipeline Prediksi

Aplikasi berjalan dengan pipeline end-to-end berikut:

```
[Foto Konjungtiva Mata]
        │
        ▼
[Flip Kiri→Kanan]  ← Jika foto sisi kiri, di-mirror agar konsisten
        │
        ▼
[Lokalisasi & Crop Konjungtiva]  ← Color thresholding + contour detection
        │
        ▼
[Resize & Pad 224×224]  ← Aspek rasio dipertahankan, pad border hitam
        │
        ▼
[Test-Time Augmentation (TTA)]  ← 8x forward passes (rotasi, flip, brightness)
        │
        ▼
[EfficientNetB0 + Dense Head (Sigmoid)]  ← Model ONNX end-to-end
        │
        ▼
[Rata-rata P(anemia) → Threshold 0.50]  ← Klasifikasi: Anemia / Non-Anemia
```

1. **Image Augmentation**: Foto sisi kiri di-flip horizontal agar konsisten dengan sisi kanan.
2. **Conjunctiva Localization**: Deteksi dan crop area konjungtiva menggunakan hierarchical color thresholding (RGB+HSV) dengan masking background.
3. **Preprocessing**: Resize ke 224×224 dengan aspek rasio dipertahankan dan padding hitam.
4. **TTA (Test-Time Augmentation)**: 8 forward passes dengan variasi augmentasi untuk prediksi yang lebih robust.
5. **ONNX Inference**: Model EfficientNetB0 end-to-end (backbone + classifier head) dijalankan via ONNX Runtime. Output sigmoid menghasilkan probabilitas anemia.

---

## ⚙️ Spesifikasi & Stack Teknologi

| Komponen | Teknologi | Keterangan |
|---|---|---|
| **Backend API** | FastAPI (Python 3.10) | Asinkron, cepat, otomatis menghasilkan dokumentasi API |
| **Frontend UI** | HTML5, CSS3, React (Babel runtime) | SPA responsif dalam satu file `index.html` |
| **Database** | PostgreSQL 15 | Menyimpan histori scan dan autentikasi pengguna |
| **ML Inference** | ONNX Runtime | EfficientNetB0 end-to-end, ringan dan cepat |
| **Image Processing** | OpenCV, Pillow | Validasi, crop, dan preprocessing konjungtiva |
| **Containerization** | Docker, Docker Compose | Deployment satu perintah |

---

## 📂 Struktur Repositori

```
HemoScan/
├── server.py                   # FastAPI backend server
├── inference.py                # Pipeline prediksi ONNX + TTA
├── config.py                   # Konfigurasi terpusat (DB, paths, model)
├── image_processing.py         # Validasi & crop citra konjungtiva
├── index.html                  # Frontend SPA (React)
│
├── augmentation/               # Modul augmentasi gambar
│   ├── __init__.py
│   └── flip_to_right.py        # Flip foto kiri → kanan
│
├── database/                   # Schema & inisialisasi PostgreSQL
│   ├── init_db.py              # Script buat database + tabel
│   └── postgres_schema.sql     # DDL tabel users & scans
│
├── model_output/               # Model inferensi
│   ├── anemia_model_final.onnx # EfficientNetB0 end-to-end (ONNX)
│   └── config.pkl              # Konfigurasi threshold & parameter
│
├── assets/                     # Aset statis (logo, background, loader)
├── notebooks/                  # Jupyter notebook training + artefak
├── uploads/                    # Gambar pasien yang diunggah (gitignored)
│
├── Dockerfile                  # Konfigurasi container
├── docker-compose.yml          # Orkestrasi app + PostgreSQL
├── entrypoint.sh               # Startup script container
│
├── requirements.txt            # Dependensi runtime (server)
├── requirements.docker.txt     # Dependensi Docker (headless)
├── requirements.dev.txt        # Dependensi development/training
│
├── .env.example                # Template environment variables
├── .gitignore
├── .dockerignore
└── README.md
```

---

## ✅ Status Sistem & Model

Model inferensi yang digunakan:

| File | Keterangan |
|---|---|
| `model_output/anemia_model_final.onnx` | EfficientNetB0 + Dense Head (Sigmoid), ONNX format |
| `model_output/config.pkl` | Konfigurasi threshold (0.50), jumlah TTA (8), class mapping |

---

## 📋 Prasyarat (Prerequisites)

Sebelum menjalankan aplikasi, pastikan salah satu dari prasyarat berikut terpenuhi:

### Untuk Docker (Opsi 1 — Direkomendasikan)

| Software | Versi Minimum | Link Download |
|---|---|---|
| **Docker Desktop** | 4.0+ | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |

> **Catatan untuk macOS Apple Silicon (M1/M2/M3/M4):**
> Docker Compose sudah dikonfigurasi dengan `platform: linux/amd64`, sehingga otomatis kompatibel melalui Rosetta 2. Pastikan opsi **"Use Rosetta for x86_64/amd64 emulation on Apple Silicon"** dicentang di Docker Desktop → Settings → General.

### Untuk Lokal (Opsi 2)

| Software | Versi Minimum | Link Download |
|---|---|---|
| **Python** | 3.10+ | [python.org/downloads](https://www.python.org/downloads/) |
| **PostgreSQL** | 13+ | [postgresql.org/download](https://www.postgresql.org/download/) |
| **pip** | 21+ | Sudah termasuk dalam Python |

---

## ▶️ Cara Menjalankan Aplikasi

### OPSI 1: Docker Compose ⭐ (Direkomendasikan — Semua OS)

Cara ini **paling mudah** dan berjalan di **Windows, macOS (Intel & Apple Silicon), dan Linux** tanpa perlu install Python atau PostgreSQL secara manual.

1. **Install Docker Desktop** dari [docker.com](https://www.docker.com/products/docker-desktop/) dan pastikan sudah berjalan.

2. **Buka terminal** di folder proyek ini:

   | OS | Cara Buka Terminal |
   |---|---|
   | **Windows** | Klik kanan folder → *Open in Terminal*, atau buka PowerShell lalu `cd` ke folder proyek |
   | **macOS** | Klik kanan folder → *New Terminal at Folder*, atau buka Terminal lalu `cd` ke folder proyek |
   | **Linux** | Buka Terminal lalu `cd` ke folder proyek |

3. **Jalankan satu perintah ini:**
   ```bash
   docker compose up --build
   ```
   > ⏳ Proses pertama kali membutuhkan waktu 3–5 menit untuk mengunduh image dan build container. Selanjutnya akan lebih cepat.

4. **Tunggu** hingga muncul log:
   ```
   ================================================
      App running at: http://localhost:8000
   ================================================
   ```

5. **Buka browser** dan akses: **http://localhost:8000**

6. **Untuk mematikan**, tekan `Ctrl+C` pada terminal, lalu:
   ```bash
   docker compose down
   ```
   > Gunakan `docker compose down -v` jika ingin **menghapus semua data database** dan mulai dari awal.

---

### OPSI 2: Menjalankan Lokal (Manual)

Gunakan opsi ini jika tidak ingin menggunakan Docker. Membutuhkan **Python 3.10+** dan **PostgreSQL** yang sudah terinstall.

#### Langkah 1 — Install Python & PostgreSQL

| OS | Python | PostgreSQL |
|---|---|---|
| **Windows** | Download dari [python.org](https://www.python.org/downloads/). Centang **"Add to PATH"** saat install. | Download dari [postgresql.org](https://www.postgresql.org/download/windows/). Catat password yang diset saat install. |
| **macOS** | `brew install python@3.10` atau download dari [python.org](https://www.python.org/downloads/) | `brew install postgresql@15 && brew services start postgresql@15` |
| **Linux (Ubuntu/Debian)** | `sudo apt install python3.10 python3.10-venv python3-pip` | `sudo apt install postgresql postgresql-contrib && sudo systemctl start postgresql` |

#### Langkah 2 — Install Dependensi Python

```bash
# (Opsional tapi disarankan) Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

#### Langkah 3 — Konfigurasi Database

Salin template environment lalu isi password PostgreSQL Anda:

```bash
# Windows (PowerShell / CMD):
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```

Buka file `.env` dengan text editor dan ganti `your_password_here` dengan password PostgreSQL Anda:
```env
POSTGRES_PASSWORD=password_postgresql_anda
```

#### Langkah 4 — Inisialisasi Database

```bash
python database/init_db.py
```
> Script ini akan membuat database `hemoscan`, tabel yang diperlukan, dan akun admin default.

#### Langkah 5 — Jalankan Server

```bash
python server.py
```

#### Langkah 6 — Buka Aplikasi

Buka browser dan akses: **http://localhost:8000**

---

## 🔑 Akun Login Default (Admin)

Akun administrator default dikonfigurasi secara dinamis melalui file `.env`. Nilai default (bawaan) yang digunakan saat inisialisasi adalah:

| Role | Email | Password |
|---|---|---|
| **Admin** | Diatur via `ADMIN_EMAIL`  | Diatur via `ADMIN_PASSWORD`  |


Pengguna biasa dapat mendaftar akun baru secara mandiri dengan hak akses `user` langsung dari halaman login.

---

## 🌐 Akses Publik dengan ngrok

Untuk mengekspos aplikasi ke internet (demo/presentasi):

1. Pastikan Docker berjalan (`docker compose up --build`)
2. Buka terminal baru:
   ```bash
   ngrok http 8000
   ```
3. Bagikan URL publik yang dihasilkan (contoh: `https://xxxx.ngrok-free.app`)

> Database PostgreSQL tetap aman di internal network container.

---

## 🔧 Troubleshooting

| Masalah | Solusi |
|---|---|
| **Port 8000 sudah digunakan** | Ganti `"8000:8000"` menjadi `"8080:8000"` di `docker-compose.yml`, lalu akses `http://localhost:8080` |
| **Koneksi Database Error** | Pastikan kredensial di `.env` sesuai dengan PostgreSQL Anda. Untuk Docker, tidak perlu `.env` karena sudah diatur di `docker-compose.yml` |
| **Model tidak ditemukan** | Pastikan `model_output/anemia_model_final.onnx` ada di folder proyek |
| **Docker build gagal di MacBook** | Pastikan Rosetta diaktifkan: Docker Desktop → Settings → General → centang *"Use Rosetta for x86_64/amd64 emulation"* |
| **`pip install` error `psycopg2`** | Gunakan `psycopg2-binary` (sudah di requirements.txt). Jika masih error di macOS: `brew install libpq` |
| **Permission denied `entrypoint.sh`** | Jalankan `git update-index --chmod=+x entrypoint.sh` lalu rebuild |
| **`python` command not found** | Coba gunakan `python3` sebagai pengganti `python` (umum di macOS/Linux) |

---

## 🛠️ Development

Untuk training model atau development:

```bash
pip install -r requirements.txt -r requirements.dev.txt
jupyter notebook notebooks/model.ipynb
```

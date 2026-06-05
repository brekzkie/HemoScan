# 🚀 HEMOSCAN - ANEMIA DETECTION SYSTEM

<img width="3840" height="3840" alt="image" src="https://github.com/user-attachments/assets/650e0941-d8bb-4ad3-8754-f010a91327aa" />

HemoScan adalah solusi berbasis kecerdasan buatan (AI) untuk skrining anemia secara non-invasif melalui analisis citra konjungtiva mata menggunakan teknik *Deep Learning* dan klasifikasi *Machine Learning*.

---

## 🏗️ Arsitektur & Pipeline Prediksi

Aplikasi berjalan dengan alur pipeline sebagai berikut:

```
[Citra Konjungtiva Mata] ➔ [EfficientNetB0 (Feature Extractor)] ➔ [PCA (Dimensionality Reduction)] 
                                                                          │
                                                                   (Concatenation) ➔ [XGBoost Classifier] ➔ [Hasil Prediksi Anemia]
                                                                          │
                                                    [Data Tabular: Usia, Gender, Usia Kelompok]
```

1. **Feature Extraction**: Mengekstraksi fitur warna dan tekstur dari citra konjungtiva menggunakan model **EfficientNetB0** (pre-trained ImageNet, 1280 dimensi).
2. **Dimensionality Reduction**: Mereduksi dimensi fitur gambar dari 1280 ke komponen utama menggunakan **Principal Component Analysis (PCA)**.
3. **Tabular Feature Integration**: Menggabungkan fitur gambar tereduksi dengan fitur tabular pasien (usia, jenis kelamin, kelompok usia).
4. **Classification**: Melakukan klasifikasi akhir menggunakan **XGBoost Classifier** untuk memprediksi status Anemia / Non-Anemia beserta skor probabilitas dan tingkat risiko (*risk level*).

---

## ⚙️ Spesifikasi & Stack Teknologi

| Komponen | Spesifikasi / Library | Keterangan |
|---|---|---|
| **Backend API** | FastAPI (Python 3.10) | Cepat, asinkron, dan otomatis menghasilkan dokumentasi API |
| **Frontend UI** | HTML5, CSS3 Vanilla, React (Babel runtime) | Tampilan antarmuka laboratorium empati yang responsif |
| **Database** | PostgreSQL | Menyimpan histori pemindaian dan otentikasi pengguna |
| **ML/DL Engine** | TensorFlow (≥2.13), XGBoost (≥2.0), Scikit-Learn | Library utama inferensi kecerdasan buatan |

---

## 📂 Struktur Repositori

Repositori ini telah dirapikan ke dalam struktur berikut:

```
HemoScan/
├── assets/                     # Aset statis aplikasi (gambar, video loader, dll)
│   ├── Backgorund.png
│   ├── loading.mp4
│   ├── loading.svg
│   ├── logo1.png
│   └── logo2.png
├── database/                   # Skrip inisialisasi dan skema PostgreSQL
│   ├── database.py
│   ├── init_db.py
│   └── postgres_schema.sql
├── notebooks/                  # Jupyter notebook untuk training model
│   └── model.ipynb
├── backup/                     # Arsip berkas lokal lama / tidak terpakai
│   ├── hemoscan.db             # DB SQLite lama
│   ├── scan_history.json
│   └── scan_history_local.json
├── model_output/               # Komponen model latih (weights & pickle)
│   ├── config.pkl
│   ├── label_encoder_status.pkl
│   ├── pca.pkl
│   └── xgb_anemia.json
├── uploads/                    # Gambar konjungtiva pasien yang diunggah
├── server.py                   # FastAPI backend server
├── inference.py                # Pipeline logika prediksi
├── index.html                  # Aplikasi web frontend
├── Dockerfile                  # Konfigurasi container
├── docker-compose.yml          # Orkestrasi web app + database
├── entrypoint.sh               # Skrip startup container
├── requirements.txt            # Dependensi lokal
├── requirements.docker.txt     # Dependensi Docker
├── .env                        # Variabel lingkungan
├── .dockerignore               # Pengecualian Docker build
└── README.md                   # Panduan dokumentasi utama (ini)
```

---

## ✅ Status Sistem & Model

Berkas model inferensi telah terpasang dengan lengkap di folder `model_output/`:
* `model_output/pca.pkl` - PCA reducer
* `model_output/xgb_anemia.json` - XGBoost classifier
* `model_output/config.pkl` - Konfigurasi mapping usia & threshold
* `model_output/label_encoder_status.pkl` - Label encoder status anemia

---

## ▶️ Cara Menjalankan Aplikasi

### OPSI 1: Menggunakan Docker Compose (Sangat Direkomendasikan)

Pastikan **Docker Desktop** sudah aktif di komputer Anda.

1. Buka terminal/PowerShell di folder proyek ini dan jalankan:
   ```bash
   docker compose up --build
   ```
   *Perintah ini akan menyalakan database PostgreSQL, melakukan inisialisasi tabel otomatis, menyalin aset proyek, dan menjalankan server FastAPI.*

2. Tunggu hingga muncul log sukses dari aplikasi, kemudian buka browser Anda dan akses:
   **http://localhost:8000**

3. **Cara Mematikan:**
   Tekan `Ctrl+C` pada terminal, lalu jalankan:
   ```bash
   docker compose down
   # Gunakan 'docker compose down -v' jika ingin mereset total seluruh data database
   ```

### OPSI 2: Menjalankan Secara Lokal (Manual)

1. Pasang dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
2. Pastikan database PostgreSQL lokal Anda menyala dan konfigurasikan koneksinya pada file `.env`.
3. Inisialisasi skema database:
   ```bash
   python database/init_db.py
   ```
4. Jalankan server FastAPI:
   ```bash
   python server.py
   ```
5. Akses melalui browser di: **http://localhost:8000**

---

## 🔑 Akun Login Default (Admin)

| Role | Email | Password |
|---|---|---|
| **Admin** | `admin@hemoscan.com` | `admin123` |

*Pengguna juga dapat melakukan registrasi akun baru dengan tingkat hak akses `user` langsung dari halaman login.*

---

## 🌐 Akses Publik dengan ngrok (Layanan Gratis 1 Job)

Jika Anda ingin mengekspos aplikasi HemoScan lokal Anda ke internet agar dapat diakses oleh orang lain (misalnya untuk demo/presentasi) menggunakan akun gratis **ngrok** yang membatasi 1 tunnel aktif:

1. Pastikan aplikasi Docker telah berjalan (`docker compose up --build`).
2. Buka terminal baru di komputer Anda dan jalankan perintah tunnel ngrok ke port 8000:
   ```bash
   ngrok http 8000
   ```
3. Salin URL publik yang dihasilkan oleh ngrok (contoh: `https://xxxx.ngrok-free.app`) dan bagikan URL tersebut.
4. *Database PostgreSQL tetap berjalan dengan aman secara lokal di dalam internal network container dan tidak diekspos ke publik.*

---

## 🔧 Troubleshooting

* **Port 8000 sudah digunakan oleh aplikasi lain?**
  Ganti baris ports di `docker-compose.yml` dari `"8000:8000"` menjadi `"8080:8000"`, lalu jalankan ulang dan akses `http://localhost:8080`.
* **Koneksi Database Error?**
  Pastikan kredensial di file `.env` (atau variabel lingkungan di `docker-compose.yml`) sudah sesuai dengan setelan server database PostgreSQL Anda.

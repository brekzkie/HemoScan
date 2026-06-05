# 🚀 Panduan Menjalankan HemoScan dengan Docker

## Prasyarat

Pastikan **Docker Desktop** sudah terinstal dan berjalan di komputer Anda.

- **Windows**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **macOS**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

> Verifikasi instalasi: buka terminal dan ketik `docker --version`

---

## ▶️ Cara Menjalankan (Satu Perintah)

Buka **Terminal** (macOS) atau **PowerShell / Command Prompt** (Windows), arahkan ke folder ini, lalu jalankan:

```bash
docker compose up --build
```

> Perintah ini akan:
> 1. **Build** image Docker untuk aplikasi
> 2. **Start** database PostgreSQL
> 3. **Inisialisasi** skema dan user admin otomatis
> 4. **Start** server FastAPI

**Tunggu** hingga muncul pesan:
```
hemoscan_app  | ================================================
hemoscan_app  |    App running at: http://localhost:8000
hemoscan_app  | ================================================
```

Kemudian buka browser dan akses: **http://localhost:8000**

---

## 🔑 Akun Login Default

| Role  | Email                  | Password   |
|-------|------------------------|------------|
| Admin | admin@hemoscan.com     | admin123   |

> Anda juga bisa **mendaftar akun baru** langsung dari halaman aplikasi.

---

## 🛑 Cara Mematikan

```bash
# Ctrl+C untuk menghentikan, lalu:
docker compose down
```

Data database dan foto scan akan tersimpan permanen, dan tersedia lagi saat dijalankan ulang.

```bash
# Untuk menghapus semua data (reset total):
docker compose down -v
```

---

## 🔄 Menjalankan Ulang (Setelah Pertama Kali)

```bash
# Tanpa --build (lebih cepat, karena image sudah dibuild)
docker compose up
```

---

## ❓ Troubleshooting

**Port 8000 sudah terpakai?**
```bash
# Ganti port di docker-compose.yml baris:  "8000:8000"  → misalnya "8080:8000"
# Lalu akses http://localhost:8080
```

**Container gagal start?**
```bash
# Lihat log lebih detail:
docker compose logs app
docker compose logs db
```

**Reset total (hapus semua data):**
```bash
docker compose down -v
docker compose up --build
```

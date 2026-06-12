# ================================================================
#  image_processing.py
#  Validasi dan cropping citra konjungtiva untuk HemoScan
# ================================================================

import numpy as np
import cv2
from PIL import Image


# ─── Validasi Gambar Konjungtiva ─────────────────────────────────
def validate_conjunctiva_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Validasi apakah gambar yang diunggah terlihat seperti foto konjungtiva mata.
    Menggunakan heuristik histogram warna untuk mendeteksi jaringan kemerahan/pink.

    Returns:
        (is_valid, reason) — True jika lolos validasi, beserta pesan alasan.
    """
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total_pixels = 224 * 224

    # Check 1: Kecerahan tidak terlalu gelap / terlalu terang
    mean_brightness = arr.mean()
    if mean_brightness < 30:
        return False, "Gambar terlalu gelap. Pastikan pencahayaan cukup saat mengambil foto konjungtiva."
    if mean_brightness > 245:
        return False, "Gambar terlalu terang/putih. Pastikan kamera fokus pada area konjungtiva."

    # Check 2: Kanal merah harus dominan (ciri khas kulit/konjungtiva)
    red_dominant_pixels = np.sum((r > b + 5) & (r > 60)) / total_pixels
    if red_dominant_pixels < 0.15:
        return False, (
            "Foto yang dikirim bukan merupakan gambar konjungtiva mata. "
            "Silakan ambil ulang foto bagian dalam kelopak mata bawah."
        )

    # Check 3: Harus ada piksel jaringan kemerahan/pink
    pinkish_mask = (r > 80) & (r > g) & (g > b * 0.5) & (r - b > 15)
    pinkish_ratio = np.sum(pinkish_mask) / total_pixels
    if pinkish_ratio < 0.08:
        return False, (
            "Foto yang dikirim bukan merupakan gambar konjungtiva mata. "
            "Silakan ambil ulang foto bagian dalam kelopak mata bawah."
        )

    # Check 4: Gambar tidak boleh terlalu seragam (warna solid / kosong)
    std_r, std_g, std_b = r.std(), g.std(), b.std()
    avg_std = (std_r + std_g + std_b) / 3
    if avg_std < 8:
        return False, "Gambar terdeteksi sebagai warna solid. Silakan ambil foto konjungtiva yang jelas."

    # Check 5: Bukan grayscale murni (misal dokumen teks, foto B&W)
    channel_diff = np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()
    if channel_diff < 5:
        return False, "Gambar terdeteksi sebagai hitam-putih. Silakan kirim foto berwarna dari konjungtiva mata."

    return True, "OK"


# ─── Crop Konjungtiva ────────────────────────────────────────────
def crop_conjunctiva(pil_image: Image.Image) -> Image.Image:
    """
    Lokalisasi dan crop area konjungtiva palpebra inferior (area kemerahan/pink)
    menggunakan hierarchical color thresholding, filtering posisi vertikal,
    bounding box kontur, dan background masking (piksel non-konjungtiva → putih).

    Returns:
        PIL Image hasil crop.
    """
    pil_image = pil_image.convert("RGB")

    orig_w, orig_h = pil_image.size
    process_img = pil_image.resize((224, 224))
    arr = np.array(process_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    hsv = cv2.cvtColor(np.array(process_img), cv2.COLOR_RGB2HSV)
    _h, s, _v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Hierarki mask kemerahan dengan threshold Saturasi (S) yang berbeda
    masks = [
        # Sangat ketat — untuk gambar yang terpapar cahaya merah/warm
        ("Very Strict RGB + S>115", (r > 120) & (r > g + 30) & (r - b > 50) & (s > 115)),
        # Ketat
        ("Strict RGB + S>95", (r > 105) & (r > g + 20) & (r - b > 30) & (s > 95)),
        # Sedang
        ("Medium RGB + S>80", (r > 100) & (r > g + 15) & (r - b > 25) & (s > 80)),
        # Longgar — untuk konjungtiva pucat/anemia
        ("Loose RGB + S>60", (r > 80) & (r > g) & (g > b * 0.5) & (r - b > 15) & (s > 60)),
    ]

    for mask_name, mask_expr in masks:
        mask_uint8 = mask_expr.astype(np.uint8) * 255

        # Morphological close untuk smooth kontur dan hapus noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Filter kontur berdasarkan posisi vertikal (region mata: 30 < cy < 135)
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= 3:
                x, y, w, h_box = cv2.boundingRect(c)
                cy = y + h_box / 2.0
                if 30 < cy < 135:
                    valid_contours.append((c, area))

        if not valid_contours:
            continue

        # Ambil kontur terbesar sebagai anchor konjungtiva utama
        valid_contours = sorted(valid_contours, key=lambda vc: vc[1], reverse=True)
        largest_c = valid_contours[0][0]

        x, y, w, h_box = cv2.boundingRect(largest_c)

        # Scale bounding box ke ukuran gambar asli + padding 15%
        ymin_scaled = int(max(0, (y - h_box * 0.15) / 224 * orig_h))
        ymax_scaled = int(min(orig_h, (y + h_box + h_box * 0.15) / 224 * orig_h))
        xmin_scaled = int(max(0, (x - w * 0.15) / 224 * orig_w))
        xmax_scaled = int(min(orig_w, (x + w + w * 0.15) / 224 * orig_w))

        crop_w = xmax_scaled - xmin_scaled
        crop_h = ymax_scaled - ymin_scaled

        # Crop dari gambar asli
        cropped_pil = pil_image.crop((xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled))

        # Buat mask untuk crop area
        crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # Scale dan gambar semua kontur valid pada crop mask
        for c, _area in valid_contours:
            sc = c.copy()
            sc[:, 0, 0] = np.round(sc[:, 0, 0] / 224.0 * orig_w - xmin_scaled)
            sc[:, 0, 1] = np.round(sc[:, 0, 1] / 224.0 * orig_h - ymin_scaled)
            sc[:, 0, 0] = np.clip(sc[:, 0, 0], 0, crop_w - 1)
            sc[:, 0, 1] = np.clip(sc[:, 0, 1], 0, crop_h - 1)
            cv2.drawContours(crop_mask, [sc], -1, 255, -1)

        # Piksel di luar kontur valid → putih
        crop_np = np.array(cropped_pil)
        crop_np[crop_mask == 0] = [255, 255, 255]

        print(f"[Crop] Matched mask: {mask_name} (area={valid_contours[0][1]:.1f})")
        print(f"[Crop] Bounding box: {xmin_scaled}:{xmax_scaled}, {ymin_scaled}:{ymax_scaled}")

        return Image.fromarray(crop_np)

    # Fallback: center crop jika tidak ada region yang cocok
    print("[Crop] No suitable eye contours found. Using center crop fallback.")
    crop_size = min(orig_w, orig_h)
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    return pil_image.crop((left, top, left + crop_size, top + crop_size))

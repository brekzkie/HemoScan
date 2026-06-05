"""
flip_to_right.py
================
Upload foto sisi KIRI → otomatis diflip jadi sisi KANAN
+ siap untuk pipeline augmentasi model (resize, normalize, dsb.)

Cara pakai:
    python augmentation/flip_to_right.py --input foto_mata_kiri.png
    python augmentation/flip_to_right.py --input foto_mata_kiri.png --output hasil/
    python augmentation/flip_to_right.py --input folder_kiri/ --output folder_kanan/ --batch
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# ──────────────────────────────────────────────
# CORE: Flip kiri → kanan
# ──────────────────────────────────────────────

def flip_left_to_right(image: np.ndarray) -> np.ndarray:
    """
    Flip horizontal (mirror kiri ↔ kanan).
    Mengubah foto sisi kiri menjadi sisi kanan.
    """
    return cv2.flip(image, 1)  # flipCode=1 → horizontal flip


# ──────────────────────────────────────────────
# AUGMENTASI SIAP PAKAI (opsional, aktifkan sesuai kebutuhan)
# ──────────────────────────────────────────────

def augment_for_model(
    image: np.ndarray,
    target_size: tuple = (224, 224),
    normalize: bool = True,
    brightness_range: tuple = (0.85, 1.15),
    contrast_range: tuple = (0.85, 1.15),
    add_random_flip: bool = False,
    seed: int = 42,
) -> dict:
    """
    Pipeline augmentasi setelah flip.
    Mengembalikan dict berisi berbagai versi augmented.

    Returns:
        {
            "original_flipped": np.ndarray,        # hasil flip saja
            "resized":          np.ndarray,        # resize ke target_size
            "normalized":       np.ndarray,        # float32 [0..1] atau z-score
            "augmented_set":    list[np.ndarray],  # variasi augmented
        }
    """
    rng = np.random.default_rng(seed)
    results = {}

    # 1. Simpan hasil flip mentah
    results["original_flipped"] = image.copy()

    # 2. Resize
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    results["resized"] = resized

    # 3. Normalisasi [0..1]
    if normalize:
        norm = resized.astype(np.float32) / 255.0
        results["normalized"] = norm
    else:
        results["normalized"] = resized.copy()

    # 4. Augmented set (untuk training)
    augmented = []

    # a. Brightness variation
    for factor in np.linspace(brightness_range[0], brightness_range[1], 3):
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        enhanced = ImageEnhance.Brightness(pil_img).enhance(factor)
        aug = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        augmented.append(aug)

    # b. Contrast variation
    for factor in np.linspace(contrast_range[0], contrast_range[1], 3):
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        enhanced = ImageEnhance.Contrast(pil_img).enhance(factor)
        aug = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        augmented.append(aug)

    # c. Slight rotation (±5 derajat)
    h, w = resized.shape[:2]
    for angle in [-5, -2, 0, 2, 5]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # d. Random crop + resize (zoom-in simulasi)
    for _ in range(3):
        crop_frac = rng.uniform(0.85, 0.95)
        ch, cw = int(h * crop_frac), int(w * crop_frac)
        y0 = rng.integers(0, h - ch)
        x0 = rng.integers(0, w - cw)
        cropped = resized[y0:y0 + ch, x0:x0 + cw]
        aug = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        augmented.append(aug)

    results["augmented_set"] = augmented
    return results


# ──────────────────────────────────────────────
# SAVE HELPER
# ──────────────────────────────────────────────

def save_results(
    results: dict,
    output_dir: str,
    stem: str,
    save_augmented: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Simpan hasil flip utama
    flipped_path = out / f"{stem}_RIGHT.png"
    cv2.imwrite(str(flipped_path), results["original_flipped"])
    print(f"  Flipped   -> {flipped_path}")

    # Simpan resize
    resized_path = out / f"{stem}_RIGHT_resized.png"
    cv2.imwrite(str(resized_path), results["resized"])
    print(f"  Resized   -> {resized_path}")

    # Simpan augmented set
    if save_augmented:
        aug_dir = out / f"{stem}_augmented"
        aug_dir.mkdir(exist_ok=True)
        for i, aug in enumerate(results.get("augmented_set", [])):
            cv2.imwrite(str(aug_dir / f"aug_{i:03d}.png"), aug)
        print(f"  Augmented -> {aug_dir}/ ({len(results['augmented_set'])} gambar)")


# ──────────────────────────────────────────────
# PROSES SATU FILE
# ──────────────────────────────────────────────

def process_single(
    input_path: str,
    output_dir: str,
    target_size: tuple = (224, 224),
    save_augmented: bool = True,
) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    print(f"\nInput  : {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {input_path}")

    # Flip kiri -> kanan
    flipped = flip_left_to_right(img)

    # Augmentasi
    results = augment_for_model(flipped, target_size=target_size)

    # Simpan
    save_results(results, output_dir, stem=path.stem, save_augmented=save_augmented)


# ──────────────────────────────────────────────
# PROSES BATCH (FOLDER)
# ──────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def process_batch(
    input_dir: str,
    output_dir: str,
    target_size: tuple = (224, 224),
    save_augmented: bool = True,
) -> None:
    in_dir = Path(input_dir)
    files = [f for f in in_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT]
    if not files:
        print(f"Tidak ada gambar ditemukan di: {input_dir}")
        return

    print(f"\nBatch mode: {len(files)} gambar ditemukan di '{input_dir}'")
    for f in files:
        try:
            process_single(str(f), output_dir, target_size, save_augmented)
        except Exception as e:
            print(f"  Gagal memproses {f.name}: {e}")

    print(f"\nSelesai! Semua output di: {output_dir}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip foto sisi KIRI -> KANAN + augmentasi untuk training model."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path ke file gambar (atau folder jika --batch)"
    )
    parser.add_argument(
        "--output", "-o", default="output_right",
        help="Folder output (default: output_right/)"
    )
    parser.add_argument(
        "--size", "-s", type=int, nargs=2, default=[224, 224],
        metavar=("W", "H"),
        help="Ukuran resize target (default: 224 224)"
    )
    parser.add_argument(
        "--batch", "-b", action="store_true",
        help="Proses seluruh gambar dalam folder --input"
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Hanya flip saja, tanpa augmentasi tambahan"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_size = tuple(args.size)
    save_aug = not args.no_augment

    if args.batch:
        process_batch(args.input, args.output, target_size, save_aug)
    else:
        process_single(args.input, args.output, target_size, save_aug)
        print(f"\nSelesai! Output di: {args.output}/")


if __name__ == "__main__":
    main()

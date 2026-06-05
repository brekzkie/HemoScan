"""
augmentation/
=============
Modul augmentasi gambar untuk pipeline training model HemoScan.

Modul tersedia:
    - flip_to_right  : Flip foto mata kiri → kanan + pipeline augmentasi
"""

from .flip_to_right import (
    flip_left_to_right,
    augment_for_model,
    process_single,
    process_batch,
)

__all__ = [
    "flip_left_to_right",
    "augment_for_model",
    "process_single",
    "process_batch",
]

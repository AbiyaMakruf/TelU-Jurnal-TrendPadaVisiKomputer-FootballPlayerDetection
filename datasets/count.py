import os
from pathlib import Path
from collections import defaultdict

# --- Konfigurasi ---
SPLIT_ROOT = Path('./split/')  # Lokasi direktori hasil split
CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
SPLIT_NAMES = ["train", "valid", "test"]

# Inisialisasi struktur data untuk menyimpan hasil
class_counts = {split: defaultdict(int) for split in SPLIT_NAMES}

# --- Proses ---
for game_dir in SPLIT_ROOT.iterdir():
    if not game_dir.is_dir():
        continue

    for split in SPLIT_NAMES:
        label_dir = game_dir / split / "labels"
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_id = int(parts[0])
                        if 0 <= class_id < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[class_id]
                            class_counts[split][class_name] += 1
            except Exception as e:
                print(f"❌ Gagal memproses file {label_file}: {e}")

# --- Output Hasil ---
print("\n📊 Jumlah Label per Kelas dan Split:\n")
for split in SPLIT_NAMES:
    print(f"== {split.upper()} ==")
    for class_name in CLASS_NAMES:
        count = class_counts[split][class_name]
        print(f"  {class_name:<12}: {count}")
    print()

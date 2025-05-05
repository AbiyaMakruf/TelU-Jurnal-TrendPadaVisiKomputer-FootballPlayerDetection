import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Configuration ---
TARGET_DIR = Path("./split")  # Root split directory
CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
CLASS_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}
PLOT_KDE = True

def count_instances_and_sizes(label_dir):
    counter = Counter()
    bbox_sizes = defaultdict(list)
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                size = w * h * (1920 * 1080)  # px^2 scale
                counter[class_id] += 1
                bbox_sizes[class_id].append(size)
    return counter, bbox_sizes

def main():
    if not TARGET_DIR.exists():
        print(f"Target dir not found: {TARGET_DIR}")
        return

    size_map = defaultdict(lambda: defaultdict(list))

    for game_dir in TARGET_DIR.iterdir():
        if not game_dir.is_dir():
            continue
        for split in ["train", "valid", "test"]:
            label_dir = game_dir / split / "labels"
            if label_dir.is_dir():
                _, sizes = count_instances_and_sizes(label_dir)
                for cls_id, size_list in sizes.items():
                    size_map[CLASS_MAP[cls_id]][split].extend(size_list)

    if PLOT_KDE:
        plt.figure(figsize=(12, 6))
        for cls_name in CLASS_NAMES:
            all_sizes = (
                size_map[cls_name]['train'] +
                size_map[cls_name]['valid'] +
                size_map[cls_name]['test']
            )
            if all_sizes:
                sns.kdeplot(all_sizes, label=cls_name, fill=True, common_norm=False)

        plt.xlabel("Bounding Box Size (px²)")
        plt.ylabel("Density")
        plt.title("KDE Plot of Bounding Box Sizes per Class")
        plt.legend(title="Class")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
TARGET_DIR = Path("./split")  # Root split directory
CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
CLASS_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}
PLOT_HISTOGRAM = True  # Set False if you don't want to display histogram


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
                size = w * h * (1920 * 1080)  # Assume image resolution 1920x1080 for pixel^2 scale
                counter[class_id] += 1
                bbox_sizes[class_id].append(size)
    return counter, bbox_sizes


def main():
    if not TARGET_DIR.exists():
        print(f"Target dir not found: {TARGET_DIR}")
        return

    # Count map and size map structures
    count_map = {split: defaultdict(int) for split in ["train", "valid", "test"]}
    size_map = defaultdict(lambda: defaultdict(list))

    for game_dir in TARGET_DIR.iterdir():
        if not game_dir.is_dir():
            continue

        for split in ["train", "valid", "test"]:
            label_dir = game_dir / split / "labels"
            if label_dir.is_dir():
                counts, sizes = count_instances_and_sizes(label_dir)
                for cls_id, c in counts.items():
                    cls_name = CLASS_MAP[cls_id]
                    count_map[split][cls_name] += c
                for cls_id, s in sizes.items():
                    cls_name = CLASS_MAP[cls_id]
                    size_map[cls_name][split].extend(s)

    # === Display Table of Counts ===
    df = pd.DataFrame(count_map).fillna(0).astype(int)
    print("\n📊 Bounding Box Count per Class and Dataset Split:")
    print(df)

    # === Optional Histogram ===
    if PLOT_HISTOGRAM:
        plt.figure(figsize=(12, 6))
        all_means = []
        all_medians = []

        for cls_name in CLASS_NAMES:
            all_sizes = (
                size_map[cls_name]['train'] +
                size_map[cls_name]['valid'] +
                size_map[cls_name]['test']
            )
            if all_sizes:
                plt.hist(
                    all_sizes,
                    bins=100,
                    alpha=0.5,
                    label=f"{cls_name} (avg: {np.mean(all_sizes):.1f})",
                    histtype='stepfilled'
                )
                all_means.append((cls_name, np.mean(all_sizes)))
                all_medians.append((cls_name, np.median(all_sizes)))

        plt.xlabel("Bounding Box Size (px²)")
        plt.ylabel("Number of Instances")
        plt.title("Bounding Box Size Distribution per Class")
        plt.legend(title="Class (avg size)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # Print average and median size summary
        print("\n📏 Average Bounding Box Sizes:")
        for cls, avg in sorted(all_means, key=lambda x: x[1]):
            print(f"- {cls.capitalize():<12}: {avg:.2f} px² (avg)")

        print("\n📐 Median Bounding Box Sizes:")
        for cls, med in sorted(all_medians, key=lambda x: x[1]):
            print(f"- {cls.capitalize():<12}: {med:.2f} px² (median)")


if __name__ == "__main__":
    main()

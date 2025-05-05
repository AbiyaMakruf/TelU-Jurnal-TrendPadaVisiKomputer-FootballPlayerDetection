import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

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

    results = defaultdict(lambda: Counter())  # {class_name: {'train': int, 'valid': int, 'test': int}}
    size_map = defaultdict(lambda: defaultdict(list))  # {class_name: {"train": [...], "valid": [...], "test": [...]}}

    for game_dir in TARGET_DIR.iterdir():
        if not game_dir.is_dir():
            continue

        for split in ["train", "valid", "test"]:
            label_dir = game_dir / split / "labels"
            if label_dir.is_dir():
                count, sizes = count_instances_and_sizes(label_dir)
                for cls_id, c in count.items():
                    results[CLASS_MAP[cls_id]][split] += c
                    size_map[CLASS_MAP[cls_id]][split].extend(sizes[cls_id])

    # Print Results
    for cls_name in CLASS_NAMES:
        train_val = results[cls_name]
        print(f"Class {cls_name.capitalize()}: Train {train_val['train']}, Validation {train_val['valid']}, Test {train_val['test']}")

    # Optional: Plot Histogram of Bounding Box Sizes
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
                plt.hist(all_sizes, bins=100, alpha=0.5, label=f"{cls_name} (avg: {np.mean(all_sizes):.1f})", histtype='stepfilled')
                all_means.append((cls_name, np.mean(all_sizes)))
                all_medians.append((cls_name, np.median(all_sizes)))

        plt.xlabel("Bounding Box Size (px²)")
        plt.ylabel("Number of Instances")
        plt.title("Bounding Box Size Distribution per Class")
        plt.legend(title="Class (avg size)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        print("\nAverage Bounding Box Sizes:")
        for cls, avg in sorted(all_means, key=lambda x: x[1]):
            print(f"- {cls.capitalize()}: {avg:.2f} px^2 (avg)")

        print("\nMedian Bounding Box Sizes:")
        for cls, med in sorted(all_medians, key=lambda x: x[1]):
            print(f"- {cls.capitalize()}: {med:.2f} px^2 (median)")


if __name__ == "__main__":
    main()

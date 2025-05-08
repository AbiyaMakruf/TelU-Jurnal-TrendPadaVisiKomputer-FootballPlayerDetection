import os
import shutil
import glob
from pathlib import Path
import random
import cv2

SOURCE_DIR = 'tracking-2023/test/'
TARGET_DIR = './split/'
SPLIT_RATIO = 1  # 80% train, 20% valid

# Mapping per folder
ID_MAPPINGS = {
    "SNMOT-116": {
        "referee": [4,25],
        "ball": [20],
        "goalkeeper": [19]
    },
    "SNMOT-117": {
        "referee": [3,21],
        "ball": [8],
        "goalkeeper": [11]
    },
    "SNMOT-118": {
        "referee": [14,21],
        "ball": [11],
        "goalkeeper": [9]
    },
    "SNMOT-119": {
        "referee": [15,22],
        "ball": [6],
        "goalkeeper": [5]
    },
    "SNMOT-120": {
        "referee": [22,26],
        "ball": [1],
        "goalkeeper": [11, 26]
    },
    "SNMOT-121": {
        "referee": [12],
        "ball": [16],
        "goalkeeper": [23]
    },
    "SNMOT-122": {
        "referee": [18,25],
        "ball": [19],
        "goalkeeper": [17,26]
    },
    "SNMOT-123": {
        "referee": [10],
        "ball": [12,16],
        "goalkeeper": [11,25]
    },
    "SNMOT-124": {
        "referee": [2,17],
        "ball": [5],
        "goalkeeper": [24,25]
    },
    "SNMOT-125": {
        "referee": [7,9,25],
        "ball": [8],
        "goalkeeper": [1,26]
    },
    "SNMOT-126": {
        "referee": [10,14],
        "ball": [11],
        "goalkeeper": [2]
    },
    "SNMOT-127": {
        "referee": [18,21],
        "ball": [19],
        "goalkeeper": []
    },
    "SNMOT-128": {
        "referee": [5,19],
        "ball": [13],
        "goalkeeper": [12]
    },
    "SNMOT-129": {
        "referee": [20,23],
        "ball": [21],
        "goalkeeper": [11,25]
    },
    "SNMOT-130": {
        "referee": [17],
        "ball": [19],
        "goalkeeper": [18]
    },
    "SNMOT-131": {
        "referee": [19,23],
        "ball": [1],
        "goalkeeper": [2]
    },
}

# Fungsi: tentukan kelas berdasarkan track_id
def get_class_from_track_id(snmot_name, track_id):
    mapping = ID_MAPPINGS.get(snmot_name, {})
    for cls_name, ids in mapping.items():
        if track_id in ids:
            return cls_name
    return "player"  # default

def read_game_id(gameinfo_path):
    with open(gameinfo_path, 'r') as f:
        for line in f:
            if line.startswith("gameID="):
                return line.strip().split("=")[1]
    return "unknown"

def convert_gt_to_yolo_format(gt_path, img_dir, output_dir, snmot_name):
    with open(gt_path, 'r') as f:
        lines = f.readlines()

    annotations = {}

    for line in lines:
        parts = line.strip().split(',')
        frame_id, track_id = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        cls = get_class_from_track_id(snmot_name, track_id)
        class_id = ["player", "goalkeeper", "referee", "ball"].index(cls)

        annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

        if frame_id not in annotations:
            annotations[frame_id] = []
        annotations[frame_id].append(annotation_line)

    # Tulis satu file .txt per gambar
    for frame_id, lines in annotations.items():
        label_file = os.path.join(output_dir, f"{frame_id:06d}.txt")
        with open(label_file, 'w') as f:
            f.writelines(lines)

def process_sequence(seq_path):
    gameinfo_path = os.path.join(seq_path, 'gameinfo.ini')
    img_dir = os.path.join(seq_path, 'img1')
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    snmot_name = os.path.basename(seq_path)

    if not os.path.exists(gameinfo_path) or not os.path.exists(gt_path):
        print(f"Skipping {snmot_name} (missing gameinfo.ini or gt.txt)")
        return

    game_id = read_game_id(gameinfo_path)
    target_seq_dir = os.path.join(TARGET_DIR, game_id)
    train_dir = os.path.join(target_seq_dir, 'test', 'images')
    valid_dir = os.path.join(target_seq_dir, 'valid', 'images')
    train_label_dir = os.path.join(target_seq_dir, 'test', 'labels')
    valid_label_dir = os.path.join(target_seq_dir, 'valid', 'labels')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)

    shutil.copy(gameinfo_path, target_seq_dir)

    img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    random.seed(42)
    random.shuffle(img_list)

    split_idx = int(len(img_list) * SPLIT_RATIO)
    train_imgs = img_list[:split_idx]
    valid_imgs = img_list[split_idx:]

    # Copy images
    for img in train_imgs:
        shutil.copy(img, train_dir)
    for img in valid_imgs:
        shutil.copy(img, valid_dir)

    # Konversi semua label
    convert_gt_to_yolo_format(gt_path, img_dir, train_label_dir, snmot_name)
    convert_gt_to_yolo_format(gt_path, img_dir, valid_label_dir, snmot_name)

    print(f"✅ Processed {snmot_name} → gameID {game_id}")

def main():
    Path(TARGET_DIR).mkdir(exist_ok=True)

    for snmot_folder in os.listdir(SOURCE_DIR):
        seq_path = os.path.join(SOURCE_DIR, snmot_folder)
        if os.path.isdir(seq_path):
            process_sequence(seq_path)

if __name__ == "__main__":
    main()

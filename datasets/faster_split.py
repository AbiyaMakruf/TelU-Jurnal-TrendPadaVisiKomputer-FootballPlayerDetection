import os
import shutil
import glob
from pathlib import Path
import random
import cv2
import time
from collections import defaultdict

# --- Configuration ---
SOURCE_DIR = Path('tracking-2023/test/') # Use Path objects
TARGET_DIR = Path('./split/')   # Use a new target dir name
SPLIT_RATIO = 1.0 # Set to 1.0 for test set only, < 1.0 for train/valid split

# Mapping per folder - unchanged
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
        "referee": [22,16], 
        "ball": [1], 
        "goalkeeper": [11,26]
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
    }
}

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- Helper Functions ---

# Precompute reverse mappings - unchanged
REVERSE_ID_MAPPINGS = {}
for snmot, mapping in ID_MAPPINGS.items():
    reverse_map = {}
    # Priority: Ball > Goalkeeper > Referee > Player (adjust index check if needed)
    class_priority = {"ball": 3, "referee": 2, "goalkeeper": 1, "player": 0} # Lower index = higher priority
    for cls_name, ids in mapping.items():
        for track_id in ids:
            current_priority = class_priority.get(cls_name, 0)
            existing_cls = reverse_map.get(track_id)
            existing_priority = class_priority.get(existing_cls, -1) # -1 if not present

            if current_priority > existing_priority:
                 reverse_map[track_id] = cls_name
    REVERSE_ID_MAPPINGS[snmot] = reverse_map


def get_class_from_track_id_fast(snmot_name, track_id):
    """Faster class lookup using precomputed reverse mapping."""
    cls_name = REVERSE_ID_MAPPINGS.get(snmot_name, {}).get(track_id, "player")
    return CLASS_MAP.get(cls_name, CLASS_MAP["player"]) # Return class index directly

def read_game_id(gameinfo_path):
    """Reads gameID from gameinfo.ini."""
    try:
        with open(gameinfo_path, 'r') as f:
            for line in f:
                if line.startswith("gameID="):
                    return line.strip().split("=")[1]
    except FileNotFoundError:
        print(f"Warning: gameinfo.ini not found at {gameinfo_path}")
    return "unknown"

# --- Modified Functions ---

def generate_yolo_labels(gt_path, img_dir, output_label_dir, snmot_name):
    """
    Generates all YOLO label files for a sequence efficiently.
    Prepends snmot_name to the output label filenames.
    Reads each image only once to get dimensions.
    Writes all labels to the specified output directory.
    """
    print(f"  Generating labels for {snmot_name}...")
    image_dimensions_cache = {} # Cache image dimensions: {frame_id: (w, h) or None}
    annotations_by_frame = defaultdict(list) # {frame_id: [yolo_line1, yolo_line2,...]}

    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  Error: Ground truth file not found: {gt_path}")
        return 0 # Return count of generated files

    # --- Pass 1: Aggregate annotations and cache image dimensions ---
    for line in lines:
        try:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            if w <= 0 or h <= 0: continue
        except (ValueError, IndexError): continue

        # Get image dimensions (use cache)
        img_w, img_h = -1, -1
        if frame_id in image_dimensions_cache:
            cached_dims = image_dimensions_cache[frame_id]
            if cached_dims: img_w, img_h = cached_dims
            else: continue
        else:
            img_path = img_dir / f"{frame_id:06d}.jpg"
            if not img_path.is_file():
                image_dimensions_cache[frame_id] = None; continue
            img = cv2.imread(str(img_path))
            if img is None:
                image_dimensions_cache[frame_id] = None; continue
            img_h, img_w = img.shape[:2]
            image_dimensions_cache[frame_id] = (img_w, img_h)

        # Convert to YOLO format
        if img_w <= 0 or img_h <= 0: continue
        x_center = min(max(0.0, (x + w / 2) / img_w), 1.0)
        y_center = min(max(0.0, (y + h / 2) / img_h), 1.0)
        w_norm = min(max(0.0, w / img_w), 1.0)
        h_norm = min(max(0.0, h / img_h), 1.0)

        class_id = get_class_from_track_id_fast(snmot_name, track_id)
        annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        annotations_by_frame[frame_id].append(annotation_line)

    # --- Pass 2: Write label files with prefixed names ---
    output_label_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    written_count = 0
    for frame_id, lines_to_write in annotations_by_frame.items():
        # <<< CHANGE: Add prefix to label filename >>>
        prefixed_label_name = f"{snmot_name}_{frame_id:06d}.txt"
        label_file = output_label_dir / prefixed_label_name
        try:
            with open(label_file, 'w') as f:
                f.writelines(lines_to_write)
            written_count += 1
        except IOError as e:
            print(f"  Error writing label file {label_file}: {e}")

    print(f"  Generated {written_count} label files for {len(annotations_by_frame)} frames.")
    return written_count # Return the count


def process_sequence(seq_path: Path):
    """Processes a single SNMOT sequence folder."""
    start_time = time.time()
    snmot_name = seq_path.name
    print(f"Processing {snmot_name}...")

    gameinfo_path = seq_path / 'gameinfo.ini'
    img_dir = seq_path / 'img1'
    gt_path = seq_path / 'gt' / 'gt.txt'

    # Basic checks
    if not gameinfo_path.is_file(): print(f"  Skipping {snmot_name} (missing gameinfo.ini)"); return
    if not gt_path.is_file(): print(f"  Skipping {snmot_name} (missing gt/gt.txt)"); return
    if not img_dir.is_dir(): print(f"  Skipping {snmot_name} (missing img1 directory)"); return

    game_id = read_game_id(gameinfo_path)
    target_seq_dir = TARGET_DIR / game_id

    # --- Define target directories based on SPLIT_RATIO ---
    is_test_only = (SPLIT_RATIO >= 1.0)

    if is_test_only:
        print("  Mode: Test set only (SPLIT_RATIO = 1.0)")
        # All content goes into 'test' directories
        output_img_dir = target_seq_dir / 'test' / 'images'
        output_label_dir = target_seq_dir / 'test' / 'labels'
        # Create directories
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  Mode: Train/Valid split (SPLIT_RATIO = {SPLIT_RATIO:.2f})")
        # Define separate train/valid directories
        train_img_dir = target_seq_dir / 'train' / 'images'
        valid_img_dir = target_seq_dir / 'valid' / 'images'
        train_label_dir = target_seq_dir / 'train' / 'labels'
        valid_label_dir = target_seq_dir / 'valid' / 'labels'
        # Create directories
        train_img_dir.mkdir(parents=True, exist_ok=True)
        valid_img_dir.mkdir(parents=True, exist_ok=True)
        train_label_dir.mkdir(parents=True, exist_ok=True) # Labels generated here first
        valid_label_dir.mkdir(parents=True, exist_ok=True) # Labels moved here later

    # Copy gameinfo.ini
    try:
        shutil.copy2(gameinfo_path, target_seq_dir / gameinfo_path.name)
    except Exception as e: print(f"  Error copying gameinfo.ini: {e}")

    # List and shuffle images
    img_list = sorted(list(img_dir.glob('*.jpg')))
    if not img_list: print(f"  Skipping {snmot_name} (no jpg images found)"); return

    random.seed(42)
    random.shuffle(img_list)

    # --- Handle Image/Label Generation and Placement ---

    if is_test_only:
        # --- Test Set Only Logic ---
        # Generate labels directly into the final 'test/labels' directory
        num_labels = generate_yolo_labels(gt_path, img_dir, output_label_dir, snmot_name)

        # Copy all images to 'test/images' with prefix
        print(f"  Copying {len(img_list)} images to test set...")
        copied_images = 0
        for img_path in img_list:
            # <<< CHANGE: Add prefix to image filename >>>
            prefixed_img_name = f"{snmot_name}_{img_path.name}"
            target_img_path = output_img_dir / prefixed_img_name
            try:
                shutil.copy(img_path, target_img_path)
                copied_images += 1
            except Exception as e:
                print(f"  Error copying test image {img_path.name}: {e}")
        print(f"  Copied {copied_images} images.")
        moved_labels = 0 # No labels are moved in test-only mode

    else:
        # --- Train/Valid Split Logic ---
        split_idx = int(len(img_list) * SPLIT_RATIO)
        train_imgs = img_list[:split_idx]
        valid_imgs = img_list[split_idx:]

        # Generate ALL labels first (into train_label_dir initially) with prefix
        num_labels = generate_yolo_labels(gt_path, img_dir, train_label_dir, snmot_name)

        # Copy Training Images with prefix
        print(f"  Copying {len(train_imgs)} train images...")
        copied_train_images = 0
        for img_path in train_imgs:
             # <<< CHANGE: Add prefix to image filename >>>
            prefixed_img_name = f"{snmot_name}_{img_path.name}"
            target_img_path = train_img_dir / prefixed_img_name
            try:
                shutil.copy(img_path, target_img_path)
                copied_train_images += 1
            except Exception as e: print(f"  Error copying train image {img_path.name}: {e}")
        print(f"  Copied {copied_train_images} train images.")


        # Copy Validation Images with prefix and track original names for label moving
        print(f"  Copying {len(valid_imgs)} valid images...")
        valid_img_original_basenames = set() # Store original name like '000001.jpg'
        copied_valid_images = 0
        for img_path in valid_imgs:
            valid_img_original_basenames.add(img_path.name) # Keep original name for lookup
            # <<< CHANGE: Add prefix to image filename >>>
            prefixed_img_name = f"{snmot_name}_{img_path.name}"
            target_img_path = valid_img_dir / prefixed_img_name
            try:
                shutil.copy(img_path, target_img_path)
                copied_valid_images += 1
            except Exception as e: print(f"  Error copying valid image {img_path.name}: {e}")
        print(f"  Copied {copied_valid_images} valid images.")


        # Move Validation Labels (which already have the prefix)
        print(f"  Moving {len(valid_img_original_basenames)} validation label files...")
        moved_labels = 0
        for original_base_name in valid_img_original_basenames:
            # <<< CHANGE: Construct the *prefixed* label name to find and move >>>
            original_label_stem = Path(original_base_name).stem # e.g., 000001
            prefixed_label_name = f"{snmot_name}_{original_label_stem}.txt" # e.g., SNMOT-116_000001.txt

            source_label_path = train_label_dir / prefixed_label_name
            target_label_path = valid_label_dir / prefixed_label_name

            if source_label_path.is_file():
                try:
                    # Use strings for shutil.move for broader compatibility
                    shutil.move(str(source_label_path), str(target_label_path))
                    moved_labels += 1
                except Exception as e:
                    print(f"  Error moving label file {prefixed_label_name}: {e}")
            # else: # Optional warning
            #    print(f"  Warning: Prefixed label file {prefixed_label_name} not found for validation image {original_base_name}")


    # --- Final Log ---
    end_time = time.time()
    duration = end_time - start_time
    if is_test_only:
        print(f"✅ Processed {snmot_name} (Test Only) → gameID {game_id} in {duration:.2f}s.")
    else:
        print(f"✅ Processed {snmot_name} (Train/Valid) → gameID {game_id} in {duration:.2f}s. Moved {moved_labels} labels to validation.")


def main():
    """Main function to orchestrate the processing."""
    overall_start_time = time.time()
    TARGET_DIR.mkdir(exist_ok=True) # Create root target directory

    if not SOURCE_DIR.is_dir():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        return

    sequences = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()]) # Sort for consistent order
    if not sequences:
        print(f"No sequence subdirectories found in {SOURCE_DIR}")
        return

    print(f"Found {len(sequences)} potential sequences in {SOURCE_DIR}.")
    print(f"Target directory: {TARGET_DIR}")
    if SPLIT_RATIO >= 1.0:
        print(f"Mode: Creating 'test' set only (SPLIT_RATIO={SPLIT_RATIO})")
    else:
        print(f"Mode: Creating 'train'/'valid' split (SPLIT_RATIO={SPLIT_RATIO})")
    print("-" * 30)

    for seq_path in sequences:
        process_sequence(seq_path)
        print("-" * 30)

    overall_end_time = time.time()
    print(f"\nProcessing complete. Total time: {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
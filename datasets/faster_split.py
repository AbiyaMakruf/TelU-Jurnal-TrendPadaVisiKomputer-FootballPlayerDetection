import os
import shutil
import glob
from pathlib import Path
import random # Keep for potential future use or removed completely if not needed, but currently not used for splitting
import cv2
import time
from collections import defaultdict

# --- Configuration ---
SOURCE_DIR = Path('tracking-2023/train/') # Use Path objects
TARGET_DIR = Path('./split/')   # Use a new target dir name

# --- Split Ratios (Must sum to 1.0) ---
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
TEST_RATIO = 0.2
assert TRAIN_RATIO + VALID_RATIO + TEST_RATIO == 1.0, "Split ratios must sum to 1.0"


# Mapping per folder - unchanged
ID_MAPPINGS = {
    "SNMOT-060": {
        "referee": [14, 17,26],
        "ball": [18],
        "goalkeeper": [22,25]
    },
    "SNMOT-061": {
        "referee": [3,4,26],
        "ball": [1],
        "goalkeeper": [2,25]
    },
        "SNMOT-062": {
        "referee": [18,23],
        "ball": [19],
        "goalkeeper": [17]
    },
        "SNMOT-063": {
        "referee": [8,18],
        "ball": [19],
        "goalkeeper": [1,25]
    },
        "SNMOT-064": {
            "referee": [21,22],
            "ball": [23],
            "goalkeeper": [24]
    },
        "SNMOT-065": {
            "referee": [22,23],
            "ball": [21],
            "goalkeeper": [24]
    },
        "SNMOT-066": {
            "referee": [1,5,25],
            "ball": [24],
            "goalkeeper": [24]
    },
        "SNMOT-067": {
            "referee": [21,20,25],
            "ball": [22],
            "goalkeeper": [19, 26]
    },
        "SNMOT-068": {
            "referee": [9,24],
            "ball": [8],
            "goalkeeper": [4]
    },
        "SNMOT-069": {
            "referee": [15,13],
            "ball": [18],
            "goalkeeper": []
    },
        "SNMOT-070": {
            "referee": [1,2],
            "ball": [12],
            "goalkeeper": [24, 25]
    },
        "SNMOT-071": {
            "referee": [18],
            "ball": [20],
            "goalkeeper": [19]
    },
        "SNMOT-072": {
            "referee": [2,24],
            "ball": [1],
            "goalkeeper": [22]
    },
        "SNMOT-073": {
            "referee": [7,17],
            "ball": [19],
            "goalkeeper": [24, 25]
    },
        "SNMOT-074": {
            "referee": [24,25,9],
            "ball": [23],
            "goalkeeper": [22]
    },
        "SNMOT-075": {
            "referee": [10,23],
            "ball": [22],
            "goalkeeper": [11]
    },
        "SNMOT-076": {
            "referee": [5,19,22],
            "ball": [25],
            "goalkeeper": [26]
    },
        "SNMOT-077": {
            "referee": [5,21],
            "ball": [20],
            "goalkeeper": [23]
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
    # Lower index in class_priority list = higher priority
    class_priority_list = ["player", "goalkeeper", "referee", "ball"]
    class_priority = {name: i for i, name in enumerate(class_priority_list)}

    # Iterate through priority list from highest to lowest
    for cls_name in reversed(class_priority_list): # Process ball, then ref, gk, player
        ids = mapping.get(cls_name, [])
        for track_id in ids:
             # Only assign if not already assigned by a higher priority class
            if track_id not in reverse_map:
                 reverse_map[track_id] = cls_name
    REVERSE_ID_MAPPINGS[snmot] = reverse_map


def get_class_from_track_id_fast(snmot_name, track_id):
    """Faster class lookup using precomputed reverse mapping."""
    # Default to "player" if snmot_name or track_id not in mapping
    cls_name = REVERSE_ID_MAPPINGS.get(snmot_name, {}).get(track_id, "player")
    # Return class index directly, default to player index if something goes wrong (shouldn't happen with map)
    return CLASS_MAP.get(cls_name, CLASS_MAP["player"])

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

def generate_yolo_labels_all_frames(gt_path, img_dir, output_label_dir, snmot_name):
    """
    Generates ALL YOLO label files for a sequence into the specified output_label_dir.
    Prepends snmot_name to the output label filenames.
    Reads each image only once to get dimensions.
    """
    print(f"  Generating labels for {snmot_name} (all frames)...")
    image_dimensions_cache = {} # Cache image dimensions: {frame_id: (w, h) or None}
    annotations_by_frame = defaultdict(list) # {frame_id: [yolo_line1, yolo_line2,...]}

    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  Error: Ground truth file not found: {gt_path}")
        return 0 # Return count of generated files

    # --- Pass 1: Aggregate annotations and cache image dimensions ---
    # We need dimensions for normalization. Iterate through images to build cache efficiently.
    img_paths = sorted(list(img_dir.glob('*.jpg')))
    if not img_paths:
        print(f"  No images found in {img_dir} for {snmot_name}. Cannot generate labels.")
        return 0

    # Build dimensions cache by reading images once
    for img_path in img_paths:
        try:
            frame_id = int(img_path.stem) # e.g., "000001" -> 1
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
                image_dimensions_cache[frame_id] = (img_w, img_h)
            else:
                print(f"  Warning: Could not read image {img_path}. Skipping for dimension cache.")
                image_dimensions_cache[frame_id] = None # Cache failure
        except ValueError:
             print(f"  Warning: Skipping non-numeric image file {img_path.name}.")
             image_dimensions_cache[img_path.stem] = None # Cache failure by stem name if not numeric

    # Now process GT lines using the cache
    for line in lines:
        try:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            if w <= 0 or h <= 0: continue # Skip invalid bounding boxes
        except (ValueError, IndexError): continue

        # Get image dimensions from cache
        cached_dims = image_dimensions_cache.get(frame_id)
        if not cached_dims:
             # print(f"  Warning: Dimensions for frame {frame_id:06d} not in cache or failed to read. Skipping annotation.")
             continue # Skip annotation if image dimensions aren't available

        img_w, img_h = cached_dims

        # Convert to YOLO format
        # Ensure coordinates and dimensions are within [0, 1] and valid
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Clamp values to the valid range [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        # Skip if the box collapsed or became invalid after normalization
        if w_norm <= 0 or h_norm <= 0:
            # print(f"  Warning: Normalized box invalid for frame {frame_id}, track {track_id}: w={w_norm:.4f}, h={h_norm:.4f}")
            continue


        class_id = get_class_from_track_id_fast(snmot_name, track_id)
        annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        annotations_by_frame[frame_id].append(annotation_line)

    # --- Pass 2: Write label files with prefixed names ---
    output_label_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    written_count = 0
    # Write labels only for frames that had valid annotations AND had their dimensions cached
    frames_with_annotations = sorted(annotations_by_frame.keys()) # Sort for consistent order
    for frame_id in frames_with_annotations:
        lines_to_write = annotations_by_frame[frame_id]
        # <<< CHANGE: Add prefix to label filename >>>
        prefixed_label_name = f"{snmot_name}_{frame_id:06d}.txt"
        label_file = output_label_dir / prefixed_label_name
        try:
            with open(label_file, 'w') as f:
                f.writelines(lines_to_write)
            written_count += 1
        except IOError as e:
            print(f"  Error writing label file {label_file}: {e}")

    print(f"  Generated {written_count} label files for {len(annotations_by_frame)} frames with annotations.")
    return written_count # Return the count


def process_sequence(seq_path: Path):
    """Processes a single SNMOT sequence folder with 70/10/20 chronological split."""
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

    # --- Define target directories for 3-way split ---
    train_img_dir = target_seq_dir / 'train' / 'images'
    valid_img_dir = target_seq_dir / 'valid' / 'images'
    test_img_dir = target_seq_dir / 'test' / 'images'

    # We generate all labels initially into the train label directory, then move valid/test
    train_label_dir = target_seq_dir / 'train' / 'labels'
    valid_label_dir = target_seq_dir / 'valid' / 'labels'
    test_label_dir = target_seq_dir / 'test' / 'labels'

    # Create directories
    train_img_dir.mkdir(parents=True, exist_ok=True)
    valid_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    valid_label_dir.mkdir(parents=True, exist_ok=True)
    test_label_dir.mkdir(parents=True, exist_ok=True)


    # Copy gameinfo.ini
    try:
        shutil.copy2(gameinfo_path, target_seq_dir / gameinfo_path.name)
    except Exception as e: print(f"  Error copying gameinfo.ini: {e}")

    # List images and SORT chronologically (remove random.shuffle)
    img_list = sorted(list(img_dir.glob('*.jpg')))
    if not img_list: print(f"  Skipping {snmot_name} (no jpg images found)"); return

    total_frames = len(img_list)
    num_train = int(total_frames * TRAIN_RATIO)
    num_valid = int(total_frames * VALID_RATIO)
    num_test = total_frames - num_train - num_valid # Ensure exact total

    # Divide image list chronologically
    train_imgs = img_list[:num_train]
    valid_imgs = img_list[num_train : num_train + num_valid]
    test_imgs = img_list[num_train + num_valid : ]

    print(f"  Total frames: {total_frames}")
    print(f"  Train frames: {len(train_imgs)} ({num_train})")
    print(f"  Valid frames: {len(valid_imgs)} ({num_valid})")
    print(f"  Test frames:  {len(test_imgs)} ({num_test})")


    # --- Generate ALL Labels (into train_label_dir initially) ---
    # This processes the entire gt.txt and creates label files for *all* images in the sequence.
    num_generated_labels = generate_yolo_labels_all_frames(gt_path, img_dir, train_label_dir, snmot_name)
    if num_generated_labels == 0:
         print(f"  Skipping image/label distribution for {snmot_name} due to no labels generated.")
         return # Exit processing for this sequence if no labels were generated

    # --- Copy Images and Move Labels to respective directories ---

    print(f"  Distributing images and labels...")
    copied_train_images = 0
    copied_valid_images = 0
    copied_test_images = 0
    moved_valid_labels = 0
    moved_test_labels = 0

    # Process Train Set
    for img_path in train_imgs:
        prefixed_img_name = f"{snmot_name}_{img_path.name}"
        target_img_path = train_img_dir / prefixed_img_name
        # Label is already in train_label_dir with the correct prefix
        try:
            shutil.copy(img_path, target_img_path)
            copied_train_images += 1
        except Exception as e: print(f"  Error copying train image {img_path.name}: {e}")
    print(f"  Copied {copied_train_images} train images.")

    # Process Validation Set
    for img_path in valid_imgs:
        prefixed_img_name = f"{snmot_name}_{img_path.name}"
        target_img_path = valid_img_dir / prefixed_img_name
        original_label_stem = img_path.stem # e.g., "000001"
        prefixed_label_name = f"{snmot_name}_{original_label_stem}.txt"
        source_label_path = train_label_dir / prefixed_label_name
        target_label_path = valid_label_dir / prefixed_label_name

        try:
            shutil.copy(img_path, target_img_path)
            copied_valid_images += 1
        except Exception as e: print(f"  Error copying valid image {img_path.name}: {e}")

        if source_label_path.is_file():
            try:
                shutil.move(str(source_label_path), str(target_label_path)) # Use strings for move
                moved_valid_labels += 1
            except Exception as e: print(f"  Error moving valid label {prefixed_label_name}: {e}")
        # else: # Optional warning if a label is missing for a frame
        #    print(f"  Warning: Label file {prefixed_label_name} not found for valid image {img_path.name}")

    print(f"  Copied {copied_valid_images} valid images, moved {moved_valid_labels} valid labels.")

    # Process Test Set
    for img_path in test_imgs:
        prefixed_img_name = f"{snmot_name}_{img_path.name}"
        target_img_path = test_img_dir / prefixed_img_name
        original_label_stem = img_path.stem # e.g., "000001"
        prefixed_label_name = f"{snmot_name}_{original_label_stem}.txt"
        source_label_path = train_label_dir / prefixed_label_name
        target_label_path = test_label_dir / prefixed_label_name

        try:
            shutil.copy(img_path, target_img_path)
            copied_test_images += 1
        except Exception as e: print(f"  Error copying test image {img_path.name}: {e}")

        if source_label_path.is_file():
            try:
                shutil.move(str(source_label_path), str(target_label_path)) # Use strings for move
                moved_test_labels += 1
            except Exception as e: print(f"  Error moving test label {prefixed_label_name}: {e}")
         # else: # Optional warning
         #    print(f"  Warning: Label file {prefixed_label_name} not found for test image {img_path.name}")

    print(f"  Copied {copied_test_images} test images, moved {moved_test_labels} test labels.")

    # --- Final Log ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ Processed {snmot_name} → gameID {game_id} in {duration:.2f}s.")
    print(f"  Train: {copied_train_images} images, {copied_train_images - moved_valid_labels - moved_test_labels} labels (remaining)") # Labels remaining in train_label_dir
    print(f"  Valid: {copied_valid_images} images, {moved_valid_labels} labels")
    print(f"  Test:  {copied_test_images} images, {moved_test_labels} labels")


def main():
    """Main function to orchestrate the processing."""
    overall_start_time = time.time()
    TARGET_DIR.mkdir(exist_ok=True) # Create root target directory

    if not SOURCE_DIR.is_dir():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        return

    sequences = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()]) # Sort for consistent order
    # Filter sequences based on ID_MAPPINGS keys if necessary, or process all found directories
    # sequences = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir() and d.name in ID_MAPPINGS])

    if not sequences:
        print(f"No sequence subdirectories found in {SOURCE_DIR}")
        return

    print(f"Found {len(sequences)} potential sequences in {SOURCE_DIR}.")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Split Ratio: Train={TRAIN_RATIO*100:.0f}%, Valid={VALID_RATIO*100:.0f}%, Test={TEST_RATIO*100:.0f}% (Chronological)")
    print("-" * 30)

    for seq_path in sequences:
        if seq_path.name not in ID_MAPPINGS:
             print(f"Skipping {seq_path.name}: Not found in ID_MAPPINGS.")
             print("-" * 30)
             continue # Skip directories not listed in ID_MAPPINGS

        process_sequence(seq_path)
        print("-" * 30)

    overall_end_time = time.time()
    print(f"\nProcessing complete. Total time: {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
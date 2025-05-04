import cv2
import os
import glob

SPLIT_BASE = 'split'
GAME_ID = '4'       # ganti sesuai nama folder gameID
SNMOT_ID = 'SNMOT-060'
SPLIT_TYPE = 'train'      # atau 'valid'
IMAGE_DIR = os.path.join(SPLIT_BASE, GAME_ID, SNMOT_ID, SPLIT_TYPE, 'images')
LABEL_DIR = os.path.join(SPLIT_BASE, GAME_ID, SNMOT_ID, SPLIT_TYPE, 'labels')

# Kelas objek
CLASS_NAMES = ['player', 'goalkeeper', 'referee', 'ball']
COLOR_MAP = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]  # warna berbeda

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 960, 540)

image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))
for idx, img_path in enumerate(image_paths):
    if idx % 20 != 0:  # tampilkan setiap 20 gambar
        continue

    label_filename = os.path.basename(img_path).replace('.jpg', '.txt')
    label_path = os.path.join(LABEL_DIR, label_filename)

    if not os.path.exists(label_path):
        print(f"⚠️ Label tidak ditemukan untuk {img_path}")
        continue

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # Baca file label
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])

        # Ubah ke koordinat pixel
        x = int((x_center - w / 2) * img_w)
        y = int((y_center - h / 2) * img_h)
        w = int(w * img_w)
        h = int(h * img_h)

        color = COLOR_MAP[class_id % len(COLOR_MAP)]
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)

        # Gambar BB dan teks
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Frame", resized_img)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()

import cv2
import os

GT_FILE = 'tracking-2023/train/SNMOT-063/gt/gt.txt'
IMG_FOLDER = 'tracking-2023/train/SNMOT-063/img1'

# Baca isi gt.txt
with open(GT_FILE, 'r') as f:
    lines = f.readlines()

annotations = {}
for line in lines:
    parts = line.strip().split(',')
    frame_id = int(parts[0])
    track_id = int(parts[1])
    x = int(float(parts[2]))
    y = int(float(parts[3]))
    w = int(float(parts[4]))
    h = int(float(parts[5]))

    if frame_id not in annotations:
        annotations[frame_id] = []

    annotations[frame_id].append({
        'track_id': track_id,
        'bbox': (x, y, w, h)
    })

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 960, 540)

for frame_id in sorted(annotations.keys()):
    if frame_id % 20 != 1:
        continue

    img_path = os.path.join(IMG_FOLDER, f"{frame_id:06d}.jpg")
    if not os.path.exists(img_path):
        print(f"❌ Gambar tidak ditemukan: {img_path}")
        continue

    img = cv2.imread(img_path)

    for ann in annotations[frame_id]:
        x, y, w, h = ann['bbox']
        track_id = ann['track_id']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {track_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize jika perlu:
    resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Frame", resized_img)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()

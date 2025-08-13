import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# Configuration
input_folder = "./images"
output_folder = "./output_manual"
split_width = 640
split_height = 640
overlap = 0.0
model_path = "best.pt"

os.makedirs(output_folder, exist_ok=True)

print(f"[INFO] Loading YOLO segmentation model from: {model_path}")
model = YOLO(model_path)

f = open("points.txt","w")

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"\n[INFO] Processing image: {filename}")
        path_to_img = os.path.join(input_folder, filename)
        img = cv2.imread(path_to_img)
        if img is None:
            print(f"[WARNING] Could not load image: {filename}")
            continue

        img_h, img_w, _ = img.shape
        print(f"[INFO] Image size: {img_w}x{img_h}")

        X_points = start_points(img_w, split_width, overlap)
        Y_points = start_points(img_h, split_height, overlap)
        print(f"[INFO] Splitting image into {len(Y_points)} rows and {len(X_points)} columns")

        full_annotated = np.zeros_like(img)
        detected_centers = []

        total_parts = len(Y_points) * len(X_points)
        part_counter = 1

        for row, i in enumerate(Y_points):
            for col, j in enumerate(X_points):
                print(f"[INFO] Predicting patch {part_counter}/{total_parts} at row={row}, col={col}")
                part_counter += 1

                split = img[i:i+split_height, j:j+split_width].copy()

                # Run segmentation prediction
                results = model.predict(source=split, conf=0.02, verbose=False, task='segment')[0]

                # Draw results on patch
                annotated = results.plot()

                # Extract masks
                masks = results.masks
                if masks is not None:
                    masks_data = masks.data.cpu().numpy()
                    for mask in masks_data:
                        ys, xs = np.where(mask > 0)
                        if len(xs) == 0 or len(ys) == 0:
                            continue
                        cx = int(xs.mean())
                        cy = int(ys.mean())
                        global_cx = j + cx
                        global_cy = i + cy
                        detected_centers.append((global_cx, global_cy))
                        print(f"[INFO] Detected mask centroid at (x, y): ({global_cx}, {global_cy})")
                        f.write(f"{global_cx}, {global_cy}\n")

                # Paste annotated patch
                full_annotated[i:i+split_height, j:j+split_width] = annotated

        # Draw all detected mask centroids on the full image
        for (x, y) in detected_centers:
            cv2.circle(full_annotated, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        # Get current timestamp (float)
        timestamp = time.time()

        # Convert timestamp to datetime object
        dt = datetime.fromtimestamp(timestamp)

        # Format datetime as string, for example: "YYYY-MM-DD HH:MM:SS"
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

        out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{formatted_time}.jpg")
        cv2.imwrite(out_path, full_annotated)
        print(f"[INFO] Saved annotated image to: {out_path}")

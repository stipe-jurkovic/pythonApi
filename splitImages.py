import cv2
import os
import numpy as np
from ultralytics import YOLO

# Configuration
input_folder = "./images"
output_folder = "./output"
split_width = 640
split_height = 640
overlap = 0.0
model_path = "best.pt"

os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model
print(f"[INFO] Loading YOLO model from: {model_path}")
model = YOLO(model_path)

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

        # Empty canvas for the full image
        full_annotated = np.zeros_like(img)

        total_parts = len(Y_points) * len(X_points)
        part_counter = 1

        for row, i in enumerate(Y_points):
            for col, j in enumerate(X_points):
                print(f"[INFO] Predicting patch {part_counter}/{total_parts} at row={row}, col={col}")
                part_counter += 1

                split = img[i:i+split_height, j:j+split_width].copy()

                # Run YOLOv8 on the patch
                results = model.predict(source=split, conf=0.02, verbose=False)[0]

                # Draw results on the patch
                annotated = results.plot()  # returns numpy array

                # Paste back the annotated patch into the final image
                full_annotated[i:i+split_height, j:j+split_width] = annotated

        # Save the final merged annotated image
        out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_annotated1.jpg")
        cv2.imwrite(out_path, full_annotated)
        print(f"[INFO] Saved annotated image to: {out_path}")

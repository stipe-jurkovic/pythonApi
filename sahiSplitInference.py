import os
import cv2
import numpy as np
from sahi.predict import get_sliced_prediction
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel

# Configuration
input_folder = "./images"
output_folder = "./output"
model_path = "best.pt"
slice_width = 640
slice_height = 640
overlap_ratio = 0.2
conf_threshold = 0.02

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Open output file for writing centroids
f = open("points.txt", "w")

# Load YOLOv8 segmentation model
print(f"[INFO] Loading YOLOv8 model from {model_path}")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.03,
    device="cuda:0",  # or 'cuda:0'
)

# Iterate through images
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        continue

    print(f"\n[INFO] Processing: {filename}")
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Could not read image: {filename}")
        continue

    # Run sliced inference
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )

    # Save annotated image
    out_img_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_annotatedImg.jpg")
    annotated_img = result.image  # PIL Image
    annotated_img_cv = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_img_path, annotated_img_cv)
    result.export_visuals(export_dir=output_folder,
    hide_labels=True,)
    print(f"[INFO] Exported annotated images to {output_folder}")


    # Extract and save mask centroids
    f.write(f"# {filename}\n")
    for obj in result.object_prediction_list:
        mask = obj.mask
        if mask is not None:
            ys, xs = np.where(mask.bool_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            cx = int(xs.mean())
            cy = int(ys.mean())
            f.write(f"{cx}, {cy}\n")
            print(f"[INFO] Detected mask centroid: ({cx}, {cy})")

f.close()
print("\n[INFO] Done. All centroids saved to 'points.txt'")

import os
import cv2
import numpy as np
from sahi.predict import get_sliced_prediction
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel

# Configuration
input_folder = "./images"
output_folder = "./output_sahi"
model_path = "bestCloverYOLOv8.pt"
slice_width = 640
slice_height = 640
overlap_ratio = 0.2
conf_threshold = 0.02

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 segmentation model
print(f"[INFO] Loading YOLOv8 model from {model_path}")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.03,
    device="cuda:0",  # or 'cuda:0'
)

# Iterate through images
filename = os.listdir(input_folder)[0]

print(f"\n[INFO] Processing: {filename}")
image_path = os.path.join(input_folder, filename)
image = cv2.imread(image_path)
if image is None:
    print(f"[WARNING] Could not read image: {filename}")

# Run sliced inference
result = get_sliced_prediction(
    image=image,
    detection_model=detection_model,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_ratio,
    overlap_width_ratio=overlap_ratio,
)
print(f"[INFO] Done")

result.export_visuals(export_dir=output_folder, text_size=1)

print("\n[INFO] Done. All centroids saved to 'points.txt'")

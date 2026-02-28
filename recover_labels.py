import os
import cv2
import numpy as np
import csv
import glob

IMG_DIR = "training_data/images"
CSV_FILE = "training_data/labels.csv"

image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_DIR, "*.png"))
print(f"Found {len(image_paths)} images. Rebuilding CSV...")

with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "roughness_score"])
    
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        

        valid_pixels = img[img != 128]
        
        if len(valid_pixels) == 0:
            continue
            

        raw_std = np.std(valid_pixels)
        clamped_std = max(4.0, min(14.0, raw_std))
        normalized_score = ((clamped_std - 4.0) / 10.0) * 100.0
        

        writer.writerow([filename, round(normalized_score, 2)])
        
        if idx % 500 == 0:
            print(f"Processed {idx} / {len(image_paths)}...")

print(f"Done! CSV fully restored with {len(image_paths)} labels.")

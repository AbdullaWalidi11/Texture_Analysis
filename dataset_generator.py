import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import glob

INPUT_DIR = "D:/PythonProjects/Texture_Task/raw_ffhq"
OUTPUT_DIR = "training_data/images"
CSV_FILE = "training_data/labels.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, output_face_blendshapes=False, 
    output_facial_transformation_matrixes=False, num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "roughness_score"])

image_paths = glob.glob(os.path.join(INPUT_DIR, "**", "*.png"), recursive=True) + glob.glob(os.path.join(INPUT_DIR, "**", "*.jpg"), recursive=True)
print(f"Found {len(image_paths)} images. Starting extraction...")

success_count = 0

for idx, img_path in enumerate(image_paths):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    
    if img is None: continue
    img = cv2.resize(img, (512, 512))
    h, w, _ = img.shape
    

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    

    if not detection_result.face_landmarks:
        print(f"[{idx}/{len(image_paths)}] Skipped {filename} (No face)")
        continue
        
    landmarks = detection_result.face_landmarks[0]
    def get_coords(indices):
        return np.array([ [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices ])

    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103]
    lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    left_eyebrow = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    right_eyebrow = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

    base_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(base_mask, [get_coords(face_oval)], 255)
    cv2.fillPoly(base_mask, [get_coords(lips)], 0); cv2.fillPoly(base_mask, [get_coords(left_eye)], 0)
    cv2.fillPoly(base_mask, [get_coords(right_eye)], 0); cv2.fillPoly(base_mask, [get_coords(left_eyebrow)], 0)
    cv2.fillPoly(base_mask, [get_coords(right_eyebrow)], 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = np.absolute(edges)
    edges = np.uint8(edges)
    hair_zones = cv2.GaussianBlur(edges, (25, 25), 0)
    _, beard_mask = cv2.threshold(hair_zones, 25, 255, cv2.THRESH_BINARY)
    dynamic_skin_mask = cv2.bitwise_not(beard_mask)
    final_mask = cv2.bitwise_and(base_mask, dynamic_skin_mask)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    blurred = cv2.GaussianBlur(enhanced_l, (21, 21), 0)
    texture_map = cv2.subtract(enhanced_l, blurred)
    texture_map = cv2.add(texture_map, 128)

    neutral_bg = np.full((h, w), 128, dtype=np.uint8)
    ai_ready_output = np.where(final_mask == 255, texture_map, neutral_bg)
    valid_skin_pixels = texture_map[final_mask == 255]
    

    if len(valid_skin_pixels) < 20000:
        print(f"[{idx}/{len(image_paths)}] Skipped {filename} (Not enough skin)")
        continue

    raw_std = np.std(valid_skin_pixels)
    clamped_std = max(4.0, min(14.0, raw_std))
    normalized_score = ((clamped_std - 4.0) / 10.0) * 100.0

    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, ai_ready_output)
    
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, round(normalized_score, 2)])
        
    success_count += 1
    if idx % 100 == 0:
        print(f"[{idx}/{len(image_paths)}] Processed successfully. Total so far: {success_count}")

print(f"Done! Successfully generated {success_count} perfect training images.")

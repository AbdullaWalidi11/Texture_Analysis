import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

MODEL_PATH = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def process_master_texture(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    img = cv2.resize(img, (512, 512))
    h, w, _ = img.shape
    

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        print("No face detected.")
        return
        
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
    cv2.fillPoly(base_mask, [get_coords(lips)], 0)
    cv2.fillPoly(base_mask, [get_coords(left_eye)], 0)
    cv2.fillPoly(base_mask, [get_coords(right_eye)], 0)
    cv2.fillPoly(base_mask, [get_coords(left_eyebrow)], 0)
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

    neutral_background = np.full((h, w), 128, dtype=np.uint8)
    ai_ready_output = np.where(final_mask == 255, texture_map, neutral_background)

    valid_skin_pixels = texture_map[final_mask == 255]
    
    if len(valid_skin_pixels) > 0:

        raw_std = np.std(valid_skin_pixels)
        

        MIN_STD = 4.0
        MAX_STD = 14.0
        

        clamped_std = max(MIN_STD, min(MAX_STD, raw_std))
        

        normalized_score = ((clamped_std - MIN_STD) / (MAX_STD - MIN_STD)) * 100.0
        

        cv2.putText(ai_ready_output, f"Roughness: {normalized_score:.1f}/100", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"Raw Math: {raw_std:.2f} -> UI Score: {normalized_score:.1f} / 100")
    else:
        print("No valid skin detected to score.")

    cv2.imshow("Original", img)
    cv2.imshow("AI Ready Output", ai_ready_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_master_texture("assets/image copy 2.png")

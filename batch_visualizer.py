import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob

TEST_DIR = "test_images"
MODEL_PATH = "texture_regressor.keras"

print("Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading MediaPipe...")
TASK_PATH = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    output_face_blendshapes=False, 
    output_facial_transformation_matrixes=False, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

image_paths = glob.glob(os.path.join(TEST_DIR, "*.*"))
if not image_paths:
    print(f"No images found in '{TEST_DIR}'. Please add some photos!")
    exit()

for img_path in image_paths:
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None: continue
    
    img = cv2.resize(img, (512, 512))
    h, w, _ = img.shape
    

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    
    if not result.face_landmarks:
        print(f"Skipping {filename} - No face detected.")
        continue
        
    landmarks = result.face_landmarks[0]
    def get_coords(indices):
        return np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])

    base_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(base_mask, [get_coords(FACE_OVAL)], 255)
    cv2.fillPoly(base_mask, [get_coords(LIPS)], 0); cv2.fillPoly(base_mask, [get_coords(LEFT_EYE)], 0)
    cv2.fillPoly(base_mask, [get_coords(RIGHT_EYE)], 0); cv2.fillPoly(base_mask, [get_coords(LEFT_EYEBROW)], 0)
    cv2.fillPoly(base_mask, [get_coords(RIGHT_EYEBROW)], 0)

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

    ai_ready_output = np.where(final_mask == 255, texture_map, 128)
    ai_input_img = cv2.cvtColor(ai_ready_output, cv2.COLOR_GRAY2RGB)
    ai_input_img = cv2.resize(ai_input_img, (224, 224))
    ai_input_img = np.expand_dims(ai_input_img, axis=0)

    prediction = model.predict(ai_input_img, verbose=0)
    score = max(0.0, min(100.0, prediction[0][0]))

    display_original = img.copy()
    

    if score < 35:
        color = (0, 255, 0)
        status = "SMOOTH"
    elif score < 65:
        color = (0, 215, 255)
        status = "MODERATE"
    else:
        color = (0, 0, 255)
        status = "SEVERE"

    cv2.putText(display_original, f"Score: {score:.1f}/100", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    cv2.putText(display_original, status, (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

    ai_display = cv2.cvtColor(ai_ready_output, cv2.COLOR_GRAY2BGR)
    cv2.putText(ai_display, "AI Vision Map", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    combined_view = np.hstack((display_original, ai_display))

    cv2.imshow("MelanoScan Texture Auditor - Press ANY KEY for next image", combined_view)
    

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

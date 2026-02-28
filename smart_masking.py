import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def create_dynamic_skin_mask(image_path):
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
    

    pure_skin = cv2.bitwise_and(img, img, mask=final_mask)

    cv2.imshow("Original", img)
    cv2.imshow("Final Pure Skin", pure_skin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

create_dynamic_skin_mask("assets/istockphoto-668664470-612x612.jpg")

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

print("Loading AI Model... (This might take a few seconds)")
MODEL_PATH = "texture_regressor.keras"
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

cap = cv2.VideoCapture(0)

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

print("Starting webcam... Press 'q' to quit.")

last_predict_time = 0
current_score = 0.0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
        

    frame = cv2.flip(frame, 1)
    

    cam_h, cam_w, _ = frame.shape
    min_dim = min(cam_h, cam_w)
    start_x = (cam_w // 2) - (min_dim // 2)
    start_y = (cam_h // 2) - (min_dim // 2)
    cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    

    img = cv2.resize(cropped_frame, (512, 512))
    display_frame = img.copy()
    h, w, _ = img.shape
    

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    
    ai_ready_output = np.full((h, w), 128, dtype=np.uint8) 
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        def get_coords(indices):
            return np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
        

        cv2.polylines(display_frame, [get_coords(FACE_OVAL)], True, (0, 255, 0), 2)
        cv2.polylines(display_frame, [get_coords(LIPS)], True, (0, 0, 255), 2)
        cv2.polylines(display_frame, [get_coords(LEFT_EYE)], True, (0, 0, 255), 2)
        cv2.polylines(display_frame, [get_coords(RIGHT_EYE)], True, (0, 0, 255), 2)
        

        base_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(base_mask, [get_coords(FACE_OVAL)], 255)
        cv2.fillPoly(base_mask, [get_coords(LIPS)], 0)
        cv2.fillPoly(base_mask, [get_coords(LEFT_EYE)], 0)
        cv2.fillPoly(base_mask, [get_coords(RIGHT_EYE)], 0)
        cv2.fillPoly(base_mask, [get_coords(LEFT_EYEBROW)], 0)
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
        

        current_time = time.time()
        if current_time - last_predict_time > 0.2:

            ai_input_img = cv2.cvtColor(ai_ready_output, cv2.COLOR_GRAY2RGB)
            ai_input_img = cv2.resize(ai_input_img, (224, 224))
            ai_input_img = np.expand_dims(ai_input_img, axis=0)
            

            prediction = model.predict(ai_input_img, verbose=0)
            current_score = max(0.0, min(100.0, prediction[0][0]))
            last_predict_time = current_time

        color = (0, 255, 0) if current_score < 35 else (0, 215, 255) if current_score < 65 else (0, 0, 255)
        cv2.putText(display_frame, f"Roughness: {current_score:.1f} / 100", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

    else:
        cv2.putText(display_frame, "Face not detected!", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Live Target", display_frame)
    cv2.imshow("AI Vision", ai_ready_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

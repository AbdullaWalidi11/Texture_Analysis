import cv2
import numpy as np

def extract_topography(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
        

    img = cv2.resize(img, (512, 512))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    blurred = cv2.GaussianBlur(enhanced_l, (21, 21), 0)
    

    texture_map = cv2.subtract(enhanced_l, blurred)
    texture_map = cv2.add(texture_map, 128)

    cv2.imshow("Original", img)
    cv2.imshow("Extracted Texture Map", texture_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

extract_topography("assets/close-up_acne_scar_1304321910.webp")

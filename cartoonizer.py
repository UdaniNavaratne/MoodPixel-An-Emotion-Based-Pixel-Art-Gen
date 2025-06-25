import cv2
import numpy as np
from PIL import Image

def cartoonize_image(img_pil):
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
               cv2.ADAPTIVE_THRESH_MEAN_C,
               cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

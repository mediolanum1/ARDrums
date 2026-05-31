import cv2
import numpy as np

class Preprocessor:
    def __init__(self): 
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        blurred_l = cv2.GaussianBlur(l, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred_l, cv2.CV_16S, ksize=3)
        sharpened_l = np.int16(l) - laplacian
        l_final = np.clip(sharpened_l, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l_final, a, b]), cv2.COLOR_LAB2BGR)

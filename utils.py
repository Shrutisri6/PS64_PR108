# utils.py
import cv2

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

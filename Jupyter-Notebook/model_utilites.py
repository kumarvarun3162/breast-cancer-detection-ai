import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_cnn_model.h5")

model = load_model(MODEL_PATH)

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224)) 
    img = img/225.0
    return np.reshape(img, (1,224,224,3))


def predict_image(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)

    if prediction > 0.5:
        print("Malignant (Cancer Detected)")
    else:
        print("Benign (No Cancer)")




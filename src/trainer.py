import cv2
import numpy as np
from PIL import Image
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
HARRCASCADE_FILE_PATH = os.path.join(BASE_DIR, "src", "harrcascade_frontalface_default.xml")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models")


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(HARRCASCADE_FILE_PATH)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(imagePath.split('-')[1])

        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(DATASET_DIR)
recognizer.train(faces, np.array(ids))

recognizer.write(os.path.join(MODEL_PATH, "trainer.yml")) 

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
import cv2
import numpy as np
import os 

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
HARRCASCADE_FILE_PATH = os.path.join(BASE_DIR, "src", "harrcascade_frontalface_default.xml")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(MODEL_PATH,'trainer.yml'))
faceCascade = cv2.CascadeClassifier(HARRCASCADE_FILE_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'Tonmoy'] 

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id, confidence)
        
        
        if (confidence > 40) and (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        elif (confidence > 0) and (confidence < 40):
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # ESC
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
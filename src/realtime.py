import numpy as np
import cv2
import os
import face_recognition

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
HARRCASCADE_FILE_PATH = os.path.join(BASE_DIR, "src", "harrcascade_frontalface_default.xml")

imgEmma_path = os.path.join(BASE_DIR, "images", "Emma Watson.jpg")
imgEmma = face_recognition.load_image_file(imgEmma_path)
imgEmma = cv2.cvtColor(imgEmma, cv2.COLOR_BGR2RGB)
encodedEmma = face_recognition.face_encodings(imgEmma)[0]

faceCascade = cv2.CascadeClassifier(HARRCASCADE_FILE_PATH)

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(10,10)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        encodedframe = face_recognition.face_encodings(frame)[0]
        result = face_recognition.compare_faces([encodedEmma], encodedframe)
        print(result)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
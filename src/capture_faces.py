import cv2
import os


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
HARRCASCADE_FILE_PATH = os.path.join(BASE_DIR, "src", "harrcascade_frontalface_default.xml")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
1

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier(HARRCASCADE_FILE_PATH)

face_id = '1'


count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        
        cv2.imwrite(os.path.join(DATASET_DIR,"User")+ "-" + str(face_id) + '-' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # ESC
    if k == 27:
        break
    elif count >= 50:
         break


cam.release()
cv2.destroyAllWindows()
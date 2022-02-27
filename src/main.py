import cv2 as cv
import face_recognition
import numpy as np
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))


def main():
    
    imgEmma_path = os.path.join(BASE_DIR, "images", "Emma Watson.jpg")
    imgEmma = face_recognition.load_image_file(imgEmma_path)
    imgEmma = cv.cvtColor(imgEmma, cv.COLOR_BGR2RGB)
    encodedEmma = face_recognition.face_encodings(imgEmma)[0]

    # face_locs = face_recognition.face_locations(imgEmma)
    # for face_loc in face_locs:
    #     cv.rectangle(imgEmma, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0,0,255),2)

    imgEmma_test_path = os.path.join(BASE_DIR, "images", "Hritthik Roshan.jpg")
    imgEmma_test = face_recognition.load_image_file(imgEmma_test_path)
    imgEmma_test = cv.cvtColor(imgEmma_test, cv.COLOR_BGR2RGB)
    encodedEmma_test = face_recognition.face_encodings(imgEmma_test)[0]

    result = face_recognition.compare_faces([encodedEmma], encodedEmma_test)
    print(result)

    
    
    # cv.imshow("Emma Watson", imgEmma)
    # cv.waitKey(0)



if __name__ == "__main__":
    main()
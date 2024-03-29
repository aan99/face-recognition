import os
import cv2
import numpy as np
import faceRecognition as fr


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("D:\ML PROJECT\TrainingData.yml")
name = {0:"Alex",1:"Ashwini",2:"Amanda"}

cap = cv2.VideoCapture(0)
while True:
    ret,test_img = cap.read()
    face_detected,gray_img = fr.faceDetection(test_img)


    for (x,y,w,h) in face_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Face Detection",resized_img)
    cv2.waitKey(10)
    for face in face_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence:", confidence)
        print("Label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]

        fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Face detection", resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import faceRecognition as fr
test_img = cv2.imread("D:\ML PROJECT\Testimg\Aaaaaa.jpg")
faces_detected, gray_img = fr.faceDetection(test_img)
print("Face Detected:", faces_detected)

'''for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow("Face detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#faces,faceID = fr.labels_for_training_data("D:\ML PROJECT\TrainingData")
#face_recognizer = fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("D:\ML PROJECT\TrainingData.yml")
name = {0:"Alex",1:"Ashwini",2:"Amanda"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print("Confidence:",confidence)
    print("Label:",label)
    fr.draw_rect(test_img,face)
    predicted_name = name[label]
    if confidence>40:
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow("Face detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
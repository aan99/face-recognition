from tkinter import *
from tkinter import messagebox
#from tester import *
#from video import *
base=Tk()
base.geometry("450x450")

def a():
    import cv2
    import os
    import numpy as np
    import faceRecognition as fr
    test_img = cv2.imread("D:\ML PROJECT\Testimg\Ma.jpg")
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("Face Detected:", faces_detected)

    '''for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow("Face detection",resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # faces,faceID = fr.labels_for_training_data("D:\ML PROJECT\TrainingData")
    # face_recognizer = fr.train_classifier(faces,faceID)
    # face_recognizer.save('trainingData.yml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("D:\ML PROJECT\TrainingData.yml")
    name = {0: "Alex", 1: "Ashwini", 2: "Amanda"}

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence:", confidence)
        print("Label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if confidence > 37:
            continue
        fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (800, 600))
    cv2.imshow("Face detection", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def b():
    import os
    import cv2
    import numpy as np
    import faceRecognition as fr

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("D:\ML PROJECT\TrainingData.yml")
    name = {0: "Alex", 1: "Ashwini", 2: "Amanda"}

    cap = cv2.VideoCapture(0)
    while True:
        ret, test_img = cap.read()
        face_detected, gray_img = fr.faceDetection(test_img)

        for (x, y, w, h) in face_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Face Detection", resized_img)
        cv2.waitKey(10)
        for face in face_detected:
            (x, y, w, h) = face
            roi_gray = gray_img[y:y + h, x:x + h]
            label, confidence = face_recognizer.predict(roi_gray)
            print("Confidence:", confidence)
            print("Label:", label)
            fr.draw_rect(test_img, face)
            predicted_name = name[label]

            if label == 1:
                messagebox.showinfo("Authentication","face detected")

            fr.put_text(test_img, predicted_name, x, y)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Face detection", resized_img)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


btn1=Button(base,text="tester",command=a)
btn1.place(x=60,y=70)
btn2=Button(base,text="camera",command=b)
btn2.place(x=200,y=70)

base.mainloop()

import cv2
import glob
import os
import time
import imutils
import argparse

subject_label = 1
font = cv2.FONT_HERSHEY_SIMPLEX
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.createLBPHFaceRecognizer()

def recognize_face(faces):
  
    predict_label = []
    predict_conf = []
    # cv2.imshow('img',faces)
    # cv2.waitKey(0)
    predict_tuple = recognizer.predict(faces)
    a, b = predict_tuple
    predict_label.append(a)
    predict_conf.append(b)
    print(predict_tuple)
    return predict_label

if __name__ == '__main__':
  
    if os.path.exists("cont.yaml"):
            recognizer.load("cont.yaml")
    img = cv2.imread('1.jpg')
    cv2.imshow('img',img)
    cv2.waitKey(0)
    img=imutils.resize(img, width=min(500, img.shape[1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]     
    label = recognize_face(roi_gray)
              
# frame_processed = put_label_on_face(frame_processed, faces, label)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(label)
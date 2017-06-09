import cv2
from matplotlib import pyplot as plt
import imutils
from skimage import feature
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')

img = cv2.imread('image.jpg')
img=imutils.resize(img, width=min(500, img.shape[1]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w] 
    features = feature.local_binary_pattern(roi_gray, 10, 5, method="uniform")
    print(len(np.unique(features)))
    print(len(features))   
 
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        roi1=gray[ey:ey+eh,ex:ex+ew]
        features1 = feature.local_binary_pattern(roi1, 10, 5, method="uniform")
        print(len(np.unique(features1)))
        print(len(features1))
        #print(features1)
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots()
        fig.suptitle("Local Binary Patterns")
        plt.ylabel("% of Pixels")
        plt.xlabel("LBP pixel bucket")
        n_bins = features1.max() + 1
        ax.hist(features1.ravel(), normed=True, bins=n_bins, range=(0,n_bins))
        plt.show()
       
    nose=nose_cascade.detectMultiScale(roi_gray)
    for (nx,ny,nw,nh) in nose:                                                          
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
        roi2=gray[ny:ny+nh,nx:nx+nw]
        features2= feature.local_binary_pattern(roi2, 10, 5, method="uniform")
        print(len(np.unique(features2)))
        print(len(features2))
        #print(features2)
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots()
        fig.suptitle("Local Binary Patterns")
        plt.ylabel("% of Pixels")
        plt.xlabel("LBP pixel bucket")
        n_bins = features2.max() + 1
        ax.hist(features2.ravel(), normed=True, bins=n_bins, range=(0,n_bins))
        plt.show()

    mouth=mouth_cascade.detectMultiScale(roi_gray)
    for (mx,my,mw,mh) in mouth:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)
        roi3=gray[my:my+mh,mx:mx+mw]
        features3 = feature.local_binary_pattern(roi3, 10, 5, method="uniform")
        print(len(np.unique(features3)))
        print(len(features3))
        #print(features3)
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots()
        fig.suptitle("Local Binary Patterns")
        plt.ylabel("% of Pixels")
        plt.xlabel("LBP pixel bucket")
        n_bins = features3.max() + 1
        ax.hist(features3.ravel(), normed=True, bins=n_bins, range=(0,n_bins))
        plt.show()
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


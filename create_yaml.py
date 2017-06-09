#!/usr/bin/python
import cv2
import os
import sys
import numpy as np
from PIL import Image
import imutils
import argparse
import glob,os
import shutil
from shutil import copyfile

cascadePath = "haarcascade_profileface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_images_and_labels():
    
    path = glob.glob('dataset/*')
    images = []
    labels = []
    count=1
    c=0
    for x in path:
	    fil="%s" %x[8:]
	    print(fil)
	    
	    for image_path in glob.glob("%s/*" %x):
	    	c+=1
	    	print(image_path)
	    	img = cv2.imread(image_path)
	    	# cv2.imshow('img',img)
	    	# cv2.waitKey(0)
	    	image = imutils.resize(img, width=min(500, img.shape[1]))
	    	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    	
	    	faces = faceCascade.detectMultiScale(image)
	    	for (x, y, w, h) in faces:
	            images.append(image[y: y + h, x: x + w])
	            # cv2.imwrite("subject02."+str(i)+".jpg",image[y: y + h, x: x + w])
	            # i=i+1
	            labels.append(count)
	           # print(labels)
	            cv2.imshow("Adding faces to traning set",
	                       image[y: y + h, x: x + w])
	            cv2.imshow('win', image[y: y + h, x: x + w])
	            cv2.waitKey(50)
	    count+=1
	    print(c) 	  
    return images, labels

images, labels = get_images_and_labels()

cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))
recognizer.save("face.yaml")

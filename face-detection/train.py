import os
import cv2
from PIL import Image
import numpy as np
from joblib import dump

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

current_id = 0
labels_id = {}

features = []
labels = []

for root, dirs, files in os.walk(image_dir):  #A loop to move through all folders in the project workspace folder
    for eachfile in files:
        if eachfile.endswith('jpg'):
            path = os.path.join(root, eachfile) #Getting the dir path of the image from root directory
            label = os.path.basename(os.path.dirname(path)).lower() #Getting names of people(folder)
            #print(label, path)

            if not label in labels_id:
                labels_id[label] = current_id
                current_id += 1
                #print(labels_id)
            
            userid = labels_id[label]

            image = Image.open(path).convert('L')
            image_array = np.array(image, 'uint8')

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.05)

            for x,y,w,h in faces:
                roi = image_array[y: y+h, x: x+w] #Region Of Interest
                features.append(roi)
                labels.append(userid)

dump(labels_id, 'face_labels.joblib')

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(features, np.array(labels))

recognizer.save('recognizer.yaml')

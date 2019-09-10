from joblib import load
import cv2

# import string
# import random

# def id_generator(size=6, chars=string.ascii_uppercase + string.digits): 
#     return ''.join(random.choice(chars) for _ in range(size))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('recognizer.yaml')

labels = load('face_labels.joblib')
labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while True:
    check, frame = cap.read()

    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grey_image, scaleFactor=1.05)

    for x,y,w,h in faces:
        roi = grey_image[y: y+h, x: x+w] 

        faceid, conf = recognizer.predict(roi) #faceid is the predicted face and conf is the confidence level of the predict

        if conf >= 50 and conf <= 100:
            cv2.putText(frame, labels[faceid], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            # if labels[faceid] == 'kwabena':
            #     print('Hello Kwabena')
            #     cap.release()
            #     cv2.destroyAllWindows()

        cv2.imshow('Real-time face recognition', frame)

        if labels[faceid] == 'Kwabena':
            cap.release()
            cv2.destroyAllWindows()

        if cv2.waitKey(1):
            break

cap.release()
cv2.destroyAllWindows()


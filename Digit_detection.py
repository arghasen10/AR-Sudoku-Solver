from keras.models import model_from_json
import json
#load json and create model
json_file = open('digitDetection1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("digitDetection1.h5")
print("Loaded model from disk")

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 100, 140, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area<=30 or area>=400:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        leng = int(h * 1.6)
        pt1 = int(y + h // 2 - leng // 2)
        pt2 = int(x + w // 2 - leng // 2)
        roi = thresh[pt1:pt1 + leng, pt2:pt2 + leng]
        if roi.shape[0] < 1 or roi.shape[1] < 1:
            continue
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        im2arr = np.array(roi)
        im2arr = im2arr.reshape(1, 1, 28, 28)
        res = classifier.predict_classes(im2arr)[0]
        cv2.putText(frame, str(res), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

def nothing(x):
    pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)
oksy = cv2.imread('oksy.png', cv2.IMREAD_UNCHANGED)

czyOksy = False
name = 0

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, +1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        clone = image

        leftXs = []
        leftYs = []
        rightXs = []
        rightYs = []

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)
        for x, y in shape[42:48]:
            rightXs.append(x)
            rightYs.append(y)

        left = clone[min(leftYs)-5:max(leftYs)+5,min(leftXs)-5:max(leftXs)+5]
        left = cv2.resize(left,(40,20))


        cv2.imshow('Left', left)
        cv2.imshow('All', image)



cap.release()
cv2.destroyAllWindows()

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
<<<<<<< HEAD
name = 1000
=======
name = 0
>>>>>>> fa2d1dacfca4071598a238c99d509ba256dc4e97

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, +1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    leftXs = []
    leftYs = []
    rightXs = []
    rightYs = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)
        for x, y in shape[42:48]:
            rightXs.append(x)
            rightYs.append(y)

<<<<<<< HEAD
    key = cv2.waitKey(1)
    if key == 27:
        break

    if leftXs.__len__()>0:
        left = image[min(leftYs) - 5:max(leftYs) + 5, min(leftXs) - 5:max(leftXs) + 5]
        left = cv2.resize(left, (40, 20))

        cv2.imshow('Left', left)
        if key == ord('q'):
            cv2.imwrite('./data/1/'+str(name)+'.jpg', left)
            print(name)
            name+=1
        elif key == ord('w'):
            cv2.imwrite('./data/2/'+str(name)+'.jpg', left)
            print(name)
            name+=1
        elif key == ord('e'):
            cv2.imwrite('./data/3/'+str(name)+'.jpg', left)
            print(name)
            name+=1
        elif key == ord('a'):
            cv2.imwrite('./data/4/'+str(name)+'.jpg', left)
            print(name)
            name+=1
        elif key == ord('s'):
            cv2.imwrite('./data/5/'+str(name)+'.jpg', left)
            print(name)
            name+=1
        elif key == ord('d'):
            cv2.imwrite('./data/6/'+str(name)+'.jpg', left)
            print(name)
            name+=1
    cv2.imshow('All', image)
=======
        left = clone[min(leftYs)-5:max(leftYs)+5,min(leftXs)-5:max(leftXs)+5]
        left = cv2.resize(left,(40,20))


        cv2.imshow('Left', left)
        cv2.imshow('All', image)
>>>>>>> fa2d1dacfca4071598a238c99d509ba256dc4e97



cap.release()
cv2.destroyAllWindows()

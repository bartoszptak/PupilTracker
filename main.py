from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)

oksy = cv2.imread('oksy.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, +1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        clone = image.copy()

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
        right = clone[min(rightYs)-5:max(rightYs)+5,min(rightXs)-5:max(rightXs)+5]

        oky, okx = oksy.shape[:2]
        oky /= 2
        okx /= 2
        
        oksy = cv2.resize(oksy,(200,200))
        b,g,r,a = cv2.split(oksy)
        ok = cv2.merge((b,g,r))
        roi = image[200:400,200:400]

        mask = cv2.medianBlur(a,3)

        img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(), mask=cv2.bitwise_not(mask))
        img2_bg = cv2.bitwise_and(ok,ok, mask=mask)

        image[200:400,200:400] = cv2.add(img1_bg,img2_bg)

        cv2.imshow('Image', image)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

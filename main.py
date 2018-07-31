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
        #left = cv2.resize(left,(40,20))
        #right = clone[min(rightYs)-5:max(rightYs)+5,min(rightXs)-5:max(rightXs)+5]

        grej = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        median = cv2.medianBlur(grej, 3)


        _, thresh = cv2.threshold(grej, 20, 255, cv2.THRESH_BINARY_INV)
        #thresh = cv2.adaptiveThreshold(grej, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
         #                          cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        im2, contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for i, x in enumerate(contours):
            area = cv2.contourArea(x)
            areas.append(area)
        if areas:
            biggestcontour = areas.index(np.max(areas))
            M = cv2.moments(contours[biggestcontour])
            #cv2.circle(left, (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), 8, (0, 0, 255), -1)
            cv2.drawContours(left, contours, biggestcontour, (0, 0, 254), 3)

        #print(contours)
        if czyOksy:
            maxL = max(rightXs)
            minR = min(leftXs)
            cenX = (maxL-minR)/2
            cenY = np.median(rightYs)
            oky, okx = oksy.shape[:2]

            b,g,r,a = cv2.split(oksy)
            ok = cv2.merge((b,g,r))
            roi = image[int(cenY-oky/2):int(cenY+oky/2),int(cenX-okx/2):int(cenX+okx/2)]

            mask = cv2.medianBlur(a, 3)

            img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
            img2_bg = cv2.bitwise_and(ok, ok, mask=mask)

            image[int(cenY - oky / 2):int(cenY + oky / 2), int(cenX - okx / 2):int(cenX + okx / 2)] = cv2.add(img1_bg,
                                                                                                         img2_bg)

        cv2.imshow('Image', image)
        cv2.imshow('Left', left)
        #cv2.imshow('Right', right)
        erode = cv2.resize(erode, None, None, 10, 10)
        cv2.imshow('Right', erode)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

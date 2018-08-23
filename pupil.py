from imutils import face_utils
import dlib
import cv2
import os

data_path = 'data'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(data_path, 'shape_predictor_68_face_landmarks.dat'))

cap = cv2.VideoCapture(0)

name = 750

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

    key = cv2.waitKey(1)
    if key == 27:
        break

    if leftXs.__len__() > 0:
        left = image[min(leftYs) - 5:max(leftYs) + 5, min(leftXs) - 5:max(leftXs) + 5]
        print(left.shape)
        left = cv2.resize(left, (200, 100))

        if key == ord('q'):
            cv2.imwrite(os.path.join(data_path, 'eyes', str(name) + '.jpg'), left)
            print(name)
            name += 1
        ra = cv2.resize(left, None, None, 2, 2)
        cv2.imshow('Left', ra)

        cv2.imshow('All', image)

cap.release()
cv2.destroyAllWindows()

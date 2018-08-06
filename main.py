from imutils import face_utils
import numpy as np
import dlib
import cv2
import ctypes
from keras.models import load_model

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model = load_model('model.h5')
model.load_weights('weights.h5')

user32 = ctypes.windll.user32
screenX, screenY = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

window_name = "Pupils Tracker"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

background = np.zeros((screenY, screenX), np.uint8)
background = np.bitwise_not(background)
background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
bg = background.copy()
cv2.imshow(window_name, bg)
cap = cv2.VideoCapture(1)

calibration = 8
pupilPositions = []

def nothing(x):
    pass


def prediction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        clone = image

        leftXs, leftYs = [],[]

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)

        lefta = clone[min(leftYs) - 5:max(leftYs) + 5, min(leftXs) - 5:max(leftXs) + 5]
        lefta = cv2.resize(lefta, (60, 30))
        left = cv2.cvtColor(lefta, cv2.COLOR_RGB2GRAY)
        left = cv2.equalizeHist(left)

        Z_ = left * 1. / 255
        Zz = np.expand_dims(Z_, axis=0)
        Z = Zz[..., np.newaxis]
        K = model.predict(Z)

        #cv2.circle(left, (K[0][0], K[0][1]), 1, (255))
        #left = cv2.resize(left, None, None, 10, 10)
        #cv2.imshow('Left', left)
        return [int(K[0][0]),int(K[0][1])]



while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, +1)

    key = cv2.waitKey(1)
    cv2.rectangle(bg, (0, 0), (int(screenX / 3), int(screenY / 2)), (0, 10, 0), -1)  # 00
    cv2.rectangle(bg, (int(screenX / 3), 0), (int(screenX / 3 * 2), int(screenY / 2)), (0, 10, 42), -1)  # 10
    cv2.rectangle(bg, (int(screenX / 3 * 2), 0), (screenX, int(screenY / 2)), (0, 10, 84), -1)  # 20

    cv2.rectangle(bg, (0, int(screenY / 2)), (int(screenX / 3), screenY), (0, 10, 126), -1)  # 01
    cv2.rectangle(bg, (int(screenX / 3), int(screenY / 2)), (int(screenX / 3 * 2), screenY), (0, 10, 168), -1)  # 11
    cv2.rectangle(bg, (int(screenX / 3 * 2), int(screenY / 2)), (screenX, screenY), (0, 10, 255), -1)  # 21


    if calibration == 8:
        cv2.putText(bg, "Cisnij spacje tej!", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2,
                    cv2.LINE_AA)
    elif calibration == 7:
        cv2.circle(bg, (int(screenX / 12), int(screenY / 8)), 10, (255, 255, 255), -1)
    elif calibration == 6:
        cv2.circle(bg, (int(screenX / 2), int(screenY / 8)), 10, (255, 255, 255), -1)
    elif calibration == 5:
        cv2.circle(bg, (int(screenX / 12 * 11), int(screenY / 8)), 10, (255, 255, 255), -1)
    elif calibration == 4:
        cv2.circle(bg, (int(screenX / 12), int(screenY / 8 * 7)), 10, (255, 255, 255), -1)
    elif calibration == 3:
        cv2.circle(bg, (int(screenX / 2), int(screenY / 8 * 7)), 10, (255, 255, 255), -1)
    elif calibration == 2:
        cv2.circle(bg, (int(screenX / 12 * 11), int(screenY / 8 * 7)), 10, (255, 255, 255), -1)
    elif calibration == 1:
        cv2.circle(bg, (int(screenX / 3), int(screenY / 2)), 10, (255, 255, 255), -1)
    elif calibration == 0:
        cv2.circle(bg, (int(screenX / 3*2), int(screenY / 2)), 10, (255, 255, 255), -1)
    else:
        bg = background.copy()
        K = prediction(image)
        print('K[0]: ',K[0],'[1][0]',pupilPositions[1][0],'[3][0]',pupilPositions[3][0])
        print('K[1]: ',K[1],'[2][1]',pupilPositions[2][1],'[5][1]',pupilPositions[5][1])

        if K[1]>pupilPositions[5][1]:
            K[1]=pupilPositions[5][1]
        elif K[1]<0:
            K[1] = pupilPositions[2][1]
        if K[0]>pupilPositions[3][0]:
            K[0]=pupilPositions[3][0]
        elif K[0]<0:
            K[0] = pupilPositions[1][0]
        x = (K[0] - pupilPositions[1][0]) / (pupilPositions[3][0] - pupilPositions[1][0])
        y = (K[1]-pupilPositions[2][1])/(pupilPositions[2][1]-pupilPositions[5][1])
        cv2.circle(bg, (int(screenX*x), int(screenY*y)), 10, (0,0,255), -1 )

    if key == 32 and calibration >= 0:
        calibration -= 1
        pupilPositions.append(prediction(image))
    elif key == 27:
        print(pupilPositions)
        break


    cv2.imshow(window_name, bg)

cap.release()
cv2.destroyAllWindows()
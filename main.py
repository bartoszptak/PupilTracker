from imutils import face_utils
import numpy as np
import dlib
import cv2
import ctypes
from keras.models import load_model
import os

data_path = 'data'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(data_path, 'shape_predictor_68_face_landmarks.dat'))

model = load_model(os.path.join(data_path, 'model.h5'))
model.load_weights(os.path.join(data_path, 'weights.h5'))

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
sightFocus = [0, 0, 0, 0, 0, 0]

bgz = cv2.imread(os.path.join(data_path, 'bg.png'))


def nothing(x):
    pass


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hsv)
    l[:] = int(255 * value)
    final_hsl = cv2.merge([h, l, s])
    img2 = cv2.cvtColor(final_hsl, cv2.COLOR_HLS2BGR)
    return img2


def distance(A, B):
    dXA = A[3] - A[1]
    dYA = A[2] - A[0]

    dXB = B[3] - B[1]
    dYB = B[2] - B[0]

    a = abs(dXB - dXA)
    b = abs(dYB - dYA)
    return (a ** 2 + b ** 2) ** (1 / 2)


def searching(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        clone = image

        leftXs, leftYs = [], []

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)

        left = clone[min(leftYs) - 5:max(leftYs) + 5, min(leftXs) - 5:max(leftXs) + 5]
        return cv2.resize(left, (100, 100))


def prediction(eye_img):
    eye_img = cv2.resize(eye_img, (100, 100))
    eye_img_gray = cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)
    eye_img_gray = cv2.equalizeHist(eye_img_gray)

    eye_img_normalized = eye_img_gray * 1. / 255
    Z1 = np.expand_dims(eye_img_normalized, axis=0)
    Z = Z1[..., np.newaxis]
    result = model.predict(Z)
    result *= 100
    return [int(result[0][0]), int(result[0][1]), int(result[0][2]), int(result[0][3])]


def averaging(a, b):
    return int((a + b) / 2)


while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, +1)

    key = cv2.waitKey(1)
    cv2.rectangle(bg, (0, 0), (int(screenX / 3), int(screenY / 2)), (0, 0, 10), -1)  # 00
    cv2.rectangle(bg, (int(screenX / 3), 0), (int(screenX / 3 * 2), int(screenY / 2)), (42, 0, 10), -1)  # 10
    cv2.rectangle(bg, (int(screenX / 3 * 2), 0), (screenX, int(screenY / 2)), (84, 0, 10), -1)  # 20

    cv2.rectangle(bg, (0, int(screenY / 2)), (int(screenX / 3), screenY), (126, 0, 10), -1)  # 01
    cv2.rectangle(bg, (int(screenX / 3), int(screenY / 2)), (int(screenX / 3 * 2), screenY), (168, 0, 10), -1)  # 11
    cv2.rectangle(bg, (int(screenX / 3 * 2), int(screenY / 2)), (screenX, screenY), (255, 0, 10), -1)  # 21

    if calibration == 8:
        cv2.putText(bg, "Cisnij spacje tej!", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2,
                    cv2.LINE_AA)
    elif calibration == 7:
        cv2.circle(bg, (int(screenX / 24), int(screenY / 16)), 10, (0, 0, 255), -1)
    elif calibration == 6:
        cv2.circle(bg, (int(screenX / 2), int(screenY / 16)), 10, (0, 0, 255), -1)
    elif calibration == 5:
        cv2.circle(bg, (int(screenX / 24 * 23), int(screenY / 16)), 10, (0, 0, 255), -1)
    elif calibration == 4:
        cv2.circle(bg, (int(screenX / 24), int(screenY / 16 * 15)), 10, (0, 0, 255), -1)
    elif calibration == 3:
        cv2.circle(bg, (int(screenX / 2), int(screenY / 16 * 15)), 10, (0, 0, 255), -1)
    elif calibration == 2:
        cv2.circle(bg, (int(screenX / 24 * 23), int(screenY / 16 * 15)), 10, (0, 0, 255), -1)
    elif calibration == 1:
        cv2.circle(bg, (int(screenX / 3), int(screenY / 2)), 10, (0, 0, 255), -1)
    elif calibration == 0:
        cv2.circle(bg, (int(screenX / 3 * 2), int(screenY / 2)), 10, (0, 0, 255), -1)
    else:
        bg = bgz.copy()
        left = searching(image)
        if left is not None:
            K = prediction(left)
            wyn = [distance(K, pupilPositions[1]),
                   distance(K, pupilPositions[2]),
                   distance(K, pupilPositions[3]),
                   distance(K, pupilPositions[4]),
                   distance(K, pupilPositions[5]),
                   distance(K, pupilPositions[6])]

            our_min = wyn.index(min(wyn))

            sightFocus[our_min] += 1

            razem = sum(sightFocus)

            reg0 = bg[0:int(screenY / 2), 0:int(screenX / 3)]
            reg1 = bg[0:int(screenY / 2), int(screenX / 3):int(screenX / 3 * 2)]
            reg2 = bg[0:int(screenY / 2), int(screenX / 3 * 2):int(screenX)]
            reg3 = bg[int(screenY / 2):int(screenY), 0:int(screenX / 3)]
            reg4 = bg[int(screenY / 2):int(screenY), int(screenX / 3):int(screenX / 3 * 2)]
            reg5 = bg[int(screenY / 2):int(screenY), int(screenX / 3 * 2):int(screenX)]

            reg0 = cv2.convertScaleAbs(reg0, alpha=sightFocus[0] / razem)
            reg1 = cv2.convertScaleAbs(reg1, alpha=sightFocus[1] / razem)
            reg2 = cv2.convertScaleAbs(reg2, alpha=sightFocus[2] / razem)
            reg3 = cv2.convertScaleAbs(reg3, alpha=sightFocus[3] / razem)
            reg4 = cv2.convertScaleAbs(reg4, alpha=sightFocus[4] / razem)
            reg5 = cv2.convertScaleAbs(reg5, alpha=sightFocus[5] / razem)

            bg[0:int(screenY / 2), 0:int(screenX / 3)] = reg0
            bg[0:int(screenY / 2), int(screenX / 3):int(screenX / 3 * 2)] = reg1
            bg[0:int(screenY / 2), int(screenX / 3 * 2):int(screenX)] = reg2
            bg[int(screenY / 2):int(screenY), 0:int(screenX / 3)] = reg3
            bg[int(screenY / 2):int(screenY), int(screenX / 3):int(screenX / 3 * 2)] = reg4
            bg[int(screenY / 2):int(screenY), int(screenX / 3 * 2):int(screenX)] = reg5

            cv2.putText(bg,
                        str(sightFocus[0]),
                        (int(screenX / 12), int(screenY / 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(bg,
                        str(sightFocus[1]),
                        (int(screenX / 2), int(screenY / 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(bg,
                        str(sightFocus[2]),
                        (int(screenX / 12 * 11), int(screenY / 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(bg,
                        str(sightFocus[3]),
                        (int(screenX / 12), int(screenY / 8 * 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(bg,
                        str(sightFocus[4]),
                        (int(screenX / 2), int(screenY / 8 * 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(bg,
                        str(sightFocus[5]),
                        (int(screenX / 12 * 11), int(screenY / 8 * 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

            cv2.circle(left, (K[0], K[1]), 2, (0, 0, 255), -2)
            cv2.circle(left, (K[2], K[3]), 2, (0, 255, 255), -2)

            # lewyX = averaging(pupilPositions[1][2], pupilPositions[4][2]) \
            #         - averaging(pupilPositions[1][0], pupilPositions[4][0])
            #
            # prawyX = averaging(pupilPositions[3][2], pupilPositions[6][2]) \
            #          - averaging(pupilPositions[3][0], pupilPositions[6][0])
            #
            # goraY = averaging(pupilPositions[1][1], pupilPositions[3][1]) \
            #         - averaging(pupilPositions[1][3], pupilPositions[3][3])
            #
            # dolY = averaging(pupilPositions[4][1], pupilPositions[6][1]) \
            #        - averaging(pupilPositions[4][3], pupilPositions[6][3])
            #
            # odlegloscX = lewyX - prawyX
            # odlegloscY = dolY - goraY
            #
            # obecnyX = K[2] - K[0] - prawyX
            # obecnyY = K[1] - K[3] - goraY
            #
            # skalaX = 1 - obecnyX / odlegloscX
            # skalaY = obecnyY / odlegloscY
            #
            # pX = int(screenX * skalaX)
            # pY = int(screenY * skalaY)
            #
            #
            #
            # si = 20
            #
            # cv2.line(bg, (pX - si, pY), (pX + si, pY), (0, 0, 255), 3)
            # cv2.line(bg, (pX, pY - si), (pX, pY + si), (0, 0, 255), 3)

            left = cv2.resize(left, None, None, 2, 2)
            bg[int(bg.shape[0] / 2 - left.shape[0] / 2):int(bg.shape[0] / 2 + left.shape[0] / 2),
            int(bg.shape[1] / 2 - left.shape[1] / 2):int(bg.shape[1] / 2 + left.shape[1] / 2)] = left

    if key == 32 and calibration >= 0:
        calibration -= 1
        pupilPositions.append(prediction(searching(image)))
    elif key == 27:
        break

    cv2.imshow(window_name, bg)

cap.release()
cv2.destroyAllWindows()

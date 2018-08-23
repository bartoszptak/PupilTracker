import cv2
import glob
import numpy as np
import os

data_path = 'data'

imgs = glob.glob(os.path.join(data_path, 'eyes', '*.jpg'))

array = []

i = 0


def transform(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        Xs.append(x)
        Xs.append(y)


cv2.namedWindow('image')
cv2.setMouseCallback('image', transform)

while i < imgs.__len__():
    Xs = []
    im = cv2.imread(imgs[i])
    im = cv2.resize(im, (100, 100))

    imz = cv2.resize(im, None, None, 5, 5)
    cv2.imshow('image', imz)
    cv2.waitKey(0)

    if len(Xs) < 4:
        continue
    array.append([im, int(Xs[0] / 5), int(Xs[1] / 5), int(Xs[2] / 5), int(Xs[3] / 5)])
    print(imgs.__len__() - i)
    i += 1

np.save(os.path.join(data_path, 'array'), array)

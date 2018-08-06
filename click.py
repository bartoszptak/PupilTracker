import cv2
import glob
import numpy as np

imgs = glob.glob('./data/*.jpg')
array = []
Xs, Ys = [0], [0]
i = 0


def transform(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        Xs[0]=x
        Ys[0]=y

cv2.namedWindow('image')
cv2.setMouseCallback('image', transform)

while i < imgs.__len__():
    im = cv2.imread(imgs[i], 0)
    im = cv2.equalizeHist(im)
    im = cv2.resize(im,None,None,1.5,1.5)
    imz = im * 1. / 255
    im = cv2.resize(im,None,None,10,10)
    cv2.imshow('image', im)
    cv2.waitKey(0)
    array.append([imz, int(Xs[0]/10), int(Ys[0]/10)])
    print(imgs.__len__()-i, ': ', int(Xs[0]/10), int(Ys[0]/10))
    i += 1

np.save('eyes', array)



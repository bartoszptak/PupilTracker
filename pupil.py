import cv2
import numpy as np


im = cv2.imread('okoxd.png')
im = cv2.resize(im,None,None,10,10)

grej = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(grej,20,255,cv2.THRESH_BINARY)

kernel = np.ones((15,15),np.uint8)
erode = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


cv2.imshow('oko', thresh)
cv2.imshow('oko2', erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
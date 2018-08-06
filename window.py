import cv2
import ctypes
import numpy as np


img = cv2.imread('data/oczy/0.jpg', cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(4.0,(4,4))
equ = clahe.apply(equ)

img = cv2.resize(img,None,None,10,10)
equ = cv2.resize(equ,None,None,10,10)
cv2.imshow('hehe',img)
cv2.imshow('equ', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

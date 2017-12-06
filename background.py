# removes background from an image


import numpy as np
import cv2


def remove_foreground(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_green = np.array([50,150,50])
    upper_green = np.array([100,255,255])
    mask = cv2.inRange(hsv,lower_green,upper_green)
    res = cv2.bitwise_and(img,img,mask=mask)
    return res


def threshold(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,25,255,cv2.THRESH_BINARY_INV)
    return thresh


frgd = cv2.imread('bgimg/img1_bg1.jpg')
mask = remove_foreground(frgd)
thresh = threshold(mask)


cv2.imwrite('bgimg/img1_thresh_bg1.jpg',thresh)







# end of file

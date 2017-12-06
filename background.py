


import numpy as np
import cv2


img = cv2.imread('bgimg/img1_bg1.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_green = np.array([50,150,50])
upper_green = np.array([255,255,255])

mask = cv2.inRange(hsv,lower_green,upper_green)
res = cv2.bitwise_and(img,img,mask=mask)

cv2.imwrite('bgimg/img1_subtract_bg1.jpg',res)










'''

import numpy as np
import cv2


def k_means(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


bkgd_sub = cv2.createBackgroundSubtractorMOG2()


frgd = cv2.imread('bgimg/img1_bg1.jpg')

img = frgd.copy()

frgd_k = k_means(frgd)
bkgd_01 = cv2.imread('bgimg/bg1.jpg')
bkgd_01_k = k_means(bkgd_01)
bkgd_02 = cv2.imread('bgimg/bg2.jpg')
bkgd_02_k = k_means(bkgd_02)

bkgd_sub.apply(bkgd_01_k,learningRate=0.5)
bkgd_sub.apply(bkgd_02_k,learningRate=0.5)
fgmask = bkgd_sub.apply(frgd_k,learningRate=0)

ret,thres_img = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(thres_img,1,np.pi/180,100,100,10)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('bgimg/img1_sub_bg1.jpg',img)

'''





'''

import numpy as np
import cv2

bkgd_sub = cv2.createBackgroundSubtractorMOG2()

frgd = cv2.imread('bgimg/img1_bg1.jpg')
bkgd = cv2.imread('bgimg/bg1.jpg')

bkgd_sub.apply(bkgd,learningRate=0.5)
#bkgd_sub.apply(bkgd_02,learningRate=0.5)
fgmask = bkgd_sub.apply(frgd,learningRate=0)

_,thresh_img = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)


cv2.imwrite('bgimg/img1_sub_bg1.jpg',thresh_img)

'''


'''

import numpy as np
import cv2


def k_means(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


bkgd_sub = cv2.createBackgroundSubtractorMOG2()


frgd = cv2.imread('bgimg/img1_bg1.jpg')
frgd_k = k_means(frgd)
bkgd_01 = cv2.imread('bgimg/bg1.jpg')
bkgd_01_k = k_means(bkgd_01)
bkgd_02 = cv2.imread('bgimg/bg2.jpg')
bkgd_02_k = k_means(bkgd_02)

bkgd_sub.apply(bkgd_01_k,learningRate=0.5)
bkgd_sub.apply(bkgd_02_k,learningRate=0.5)
fgmask = bkgd_sub.apply(frgd_k,learningRate=0)

cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)

'''






# end of file

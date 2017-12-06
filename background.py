# background subtraction

import numpy as np
import cv2


frgd = cv2.imread('bgimg/img1_bg1.jpg')

bkgd_sub = cv2.createBackgroundSubtractorMOG2()

bkgd_01 = cv2.imread('bgimg/bg1.jpg')
bkgd_02 = cv2.imread('bgimg/bg2.jpg')

bkgd_sub.apply(bkgd_01,learningRate=0.5)
bkgd_sub.apply(bkgd_02,learningRate=0.5)


fgmask = bkgd_sub.apply(frgd,learningRate=0)
cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)


# end of file

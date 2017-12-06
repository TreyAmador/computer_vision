


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


#frgd = cv2.imread('bgimg/img1_bg1.jpg')
#bkgd = cv2.imread('bgimg/bg1.jpg')
#bkgd_sub.apply(bkgd,learningRate=0.5)
#fgmask = bkgd_sub.apply(frgd,learningRate=0)

#fgmask_k = k_means(fgmask)

#cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)
#cv2.imwrite('bgimg/img1_sub_k_bg1.jpg',fgmask_k)


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




#cv2.imwrite('bgimg/img1_sub_bg1.jpg',frgd_k)

#cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)



'''

frgd = cv2.imread('bgimg/img1_bg1.jpg')

bkgd_sub = cv2.createBackgroundSubtractorMOG2()

bkgd_01 = cv2.imread('bgimg/bg1.jpg')
bkgd_02 = cv2.imread('bgimg/bg2.jpg')

bkgd_sub.apply(bkgd_01,learningRate=0.5)
bkgd_sub.apply(bkgd_02,learningRate=0.5)

fgmask = bkgd_sub.apply(frgd,learningRate=0.0)
cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)

'''










'''

import numpy as np
import cv2


bkgd_sub = cv2.createBackgroundSubtractorMOG2()

frgd = cv2.imread('bgimg/img1_bg1.jpg')
bkgd = cv2.imread('bgimg/bg1.jpg')
h,w,_ = bkgd.shape

shift_imgs = []

for i in range(-10,11):
    shift = np.float32([[1,0,i],[0,1,0]])
    shift_imgs.append(cv2.warpAffine(bkgd,shift,(w,h)))

for img in shift_imgs:
    bkgd_sub.apply(img,learningRate=0.5)

fgmask = bkgd_sub.apply(frgd,learningRate=0)
cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)

'''



'''

import numpy as np
import cv2


def quarter_image(img):
    h = img.shape[0]
    w = img.shape[1]
    q1 = img[0:h//2,w//2:w]
    q2 = img[0:h//2,0:w//2]
    q3 = img[h//2:h,0:w//2]
    q4 = img[h//2:h,w//2:w]
    return q1,q2,q3,q4


bkgd_sub = cv2.createBackgroundSubtractorMOG2()


frgd = cv2.imread('bgimg/img1_bg1.jpg')
bkgd = cv2.imread('bgimg/bg1.jpg')
qrt = list(quarter_image(bkgd))
qrt = [bkgd]


for img in qrt:
    #cv2.imshow('',cv2.resize(img,None,fx=0.25,fy=0.25))
    #cv2.waitKey(0)
    bkgd_sub.apply(img,learningRate=0.5)


fgmask = bkgd_sub.apply(frgd,learningRate=0)
cv2.imwrite('bgimg/img1_sub_bg1.jpg',fgmask)

'''


'''

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

'''

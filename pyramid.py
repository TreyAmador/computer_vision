# image blending
# this works!

import cv2
import numpy as np,sys


def init_img(filepath):
    return cv2.imread(filepath)


def gen_gaussian(orig):
    img = orig.copy()
    pyramid = [img]
    for i in range(6):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def gen_laplacian(gauss):
    pyramid = [gauss[5]]
    for i in range(5,0,-1):
        size = (gauss[i-1].shape[1],gauss[i-1].shape[0])
        GE = cv2.pyrUp(gauss[i],dstsize=size)
        laplace = cv2.subtract(gauss[i-1],GE)
        pyramid.append(laplace)
    return pyramid


def blend(laplace_a,laplace_b):
    laplace_sum = []
    for l_a,l_b in zip(laplace_a,laplace_b):
        rows,cols,dpt = l_a.shape
        l_s = np.hstack((l_a[:,0:int(cols/2)], l_b[:,int(cols/2):]))
        laplace_sum.append(l_s)
    laplace_pyramid = laplace_sum[0]
    for i in range(1,6):
        size = (laplace_sum[i].shape[1],laplace_sum[i].shape[0])
        laplace_pyramid = cv2.pyrUp(laplace_pyramid,dstsize=size)
        laplace_pyramid = cv2.add(laplace_pyramid,laplace_sum[i])
    return laplace_pyramid


def driver():
    img_a = init_img('img/apple.jpg')
    img_b = init_img('img/orange.jpg')
    gauss_a = gen_gaussian(img_a)
    gauss_b = gen_gaussian(img_b)
    laplace_a = gen_laplacian(gauss_a)
    laplace_b = gen_laplacian(gauss_b)
    blended_img = blend(laplace_a,laplace_b)
    cv2.imwrite('img/apple_orange_blended.jpg',blended_img)


if __name__ == '__main__':
    driver()



# end of file

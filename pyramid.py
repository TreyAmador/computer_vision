# image blending
# this works!
import cv2
import numpy as np,sys


def init_img(filepath):
    return cv2.imread(filepath)


def gen_gaussian(orig,depth):
    img = orig.copy()
    pyramid = [img]
    for i in range(depth+1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def gen_laplacian(gauss,depth):
    pyramid = [gauss[depth]]
    for i in range(depth,0,-1):
        size = (gauss[i-1].shape[1],gauss[i-1].shape[0])
        gauss_up = cv2.pyrUp(gauss[i],dstsize=size)
        laplace = cv2.subtract(gauss[i-1],gauss_up)
        pyramid.append(laplace)
    return pyramid


def blend_mask(laplace_a,laplace_b,gauss_mask):
    # add masking here
    laplace_sum = []
    for l_a,l_b,g_m in zip(laplace_a,laplace_b,gauss_mask):
        p_1 = (g_m/255)*l_b
        p_2 = (1-(g_m/255))*l_a
        laplace_sum.append(p_1+p_2)
    return laplace_sum

def collapse(laplace_sum,depth):
    laplace_pyramid = laplace_sum[0]
    last_pyr = laplace_sum[-1]
    for i in range(1,depth+1):
        size = (laplace_sum[i].shape[1],laplace_sum[i].shape[0])
        laplace_pyramid = cv2.pyrUp(laplace_pyramid,dstsize=size)
        laplace_pyramid = cv2.add(laplace_pyramid,laplace_sum[i])
    return laplace_pyramid


def driver():
    img_a = init_img('img/apple.jpg')
    img_b = init_img('img/orange.jpg')
    img_mask = init_img('img/mask512.jpg')
    depth = 5
    gauss_a = gen_gaussian(img_a,depth)
    gauss_b = gen_gaussian(img_b,depth)
    gauss_mask = gen_gaussian(img_mask,depth)[-2::-1]
    laplace_a = gen_laplacian(gauss_a,depth)
    laplace_b = gen_laplacian(gauss_b,depth)
    blends = blend_mask(laplace_a,laplace_b,gauss_mask)
    blended_img = collapse(blends,depth)
    cv2.imwrite('img/apple_orange_blended.jpg',blended_img)


if __name__ == '__main__':
    driver()



# end of file

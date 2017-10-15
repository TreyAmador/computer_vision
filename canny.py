# apply canny filter
from copy import deepcopy
from PIL import Image
import numpy as np
import math
import sys


def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels


def save_img(filepath,pixels):
    img = Image.fromarray(pixels)
    img.save(filepath)


def apply_canny(img):

    fltrd = gaussian_smooth(img,3)

    dx,dy = derivative_xy(fltrd)
    #save_img('img/dx.jpg',np.array(dx))
    #save_img('img/dy.jpg',np.array(dy))

    #dx_dy = sum_derivatives(dx,dy)
    #save_img('img/dx+dy.jpg',np.array(dx_dy))

    magn, origin = gradient_magn_origin(dx,dy)

    save_img('img/magnitude.jpg',np.array(magn))


    #non_max_suppress(img)
    #hysteresis_thresh(img)

    return fltrd


def gaussian_smooth(img,dim):
    off = int(dim/2)
    fltrd = deepcopy(img)
    for i in range(off,len(img)-off):
        for j in range(off,len(img[i])-off):
            fltr_sum = 0
            for s in range(i-off,i+off+1):
                for t in range(j-off,j+off+1):
                    fltr_sum += img[s][t]
            fltrd[i][j] = fltr_sum/(dim*dim)
    return fltrd


# first derivative
# left to right
# top to bottom
def derivative(img):
    drvtv = deepcopy(img)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            drvtv[i][j] = img[i+1][j] + img[i][j+1] - 2*img[i][j]
    return drvtv


def derivative_xy(img):
    dx,dy = deepcopy(img),deepcopy(img)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            dx[i][j] = img[i+1][j] - img[i][j]
            dy[i][j] = img[i][j+1] - img[i][j]
    return dx,dy


def sum_derivatives(dx,dy):
    img = deepcopy(dx)
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[i][j] = dx[i][j] + dy[i][j]
    return img


def gradient_magn_origin(dx,dy):
    magn,origin = deepcopy(dx),deepcopy(dy)
    for i in range(len(magn)):
        for j in range(len(magn[i])):
            magn[i][j] = math.sqrt(math.pow(dx[i][j],2)+math.pow(dy[i][j],2))
            origin[i][j] = math.atan2(dy[i][j],dx[i][j])
    return magn,origin


def non_max_suppress(img):
    pass


def hysteresis_thresh(img):
    pass



def driver():
    filepath = 'img/hills.jpg'
    img = init_img(filepath)
    fltrd = apply_canny(img)
    save_img('img/hills_out.jpg',fltrd)


if __name__ == '__main__':
    driver()


# medium

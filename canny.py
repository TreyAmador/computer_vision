from copy import deepcopy
from PIL import Image
import numpy as np
import math
import sys


def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath).convert('L'))
    pixels = np.array([[col for col in row] for row in img])
    return pixels


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


def derivative_xy(img):
    dx = deepcopy(img)
    dy = deepcopy(img)
    k = [1,-1]
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            dx[i][j+1] = abs(k[0]*img[i][j]+k[1]*img[i][j+1])
            dy[i+1][j] = abs(k[0]*img[i][j]+k[1]*img[i+1][j])
    return dx,dy


def gradient_magnitude(dx,dy):
    magn = deepcopy(dx)
    theta = deepcopy(dy)
    for i in range(len(dx)):
        for j in range(len(dx[0])):
            magn[i][j] = math.sqrt( \
                math.pow(dx[i][j],2) + math.pow(dy[i][j],2))
    return magn


def save_img(filepath,pixels):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    t = filepath.split('.')
    filepath = '.'.join(t[:-1])+'_canny.'+t[-1]
    img = Image.fromarray(pixels)
    img.save(filepath)


def driver():
    filepath = 'img/valve.png'
    img = init_img(filepath)
    smt = gaussian_smooth(img,3)
    dx,dy = derivative_xy(smt)

    save_img('img/valve_dx.png',dx)
    save_img('img/valve_dy.png',dy)


if __name__ == '__main__':
    driver()

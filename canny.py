from copy import deepcopy
from PIL import Image
import numpy as np
import math
import time
import sys


def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath).convert('L'))
    pixels = np.array([[col for col in row] for row in img])
    return pixels


def gen_kernels():
    kx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    ky = [[1,2,1],[0,0,0],[-1,-2,-1]]
    return kx,ky


def round_angle(rad):
    deg = (180/math.pi)*rad
    if 0 <= deg < 45:
        pass
        #print(deg,end=' ')
    elif 45 <= deg < 90:
        pass
        #print(deg,end=' ')
    elif 90 <= deg < 135:
        pass
        #print(deg,end=' ')
    elif 135 <= deg < 180:
        print(deg,end=' ')
    else:
        print('\n\nnot in value\n')
    return rad


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


# this works, but is not being using
def sobel_filter(img):
    dx = deepcopy(img)
    dy = deepcopy(img)
    kx,ky = gen_kernels()
    for i in range(len(img)-2):
        for j in range(len(img[i])-2):
            s = 0
            t = 0
            for m in range(len(kx)):
                for n in range(len(kx[m])):
                    s += dx[i+m][j+n] * kx[m][n]
                    t += dy[i+m][j+n] * ky[m][n]
            dx[i][j] = abs(s/9)
            dy[i][j] = abs(t/9)
    return dx,dy


def derivative_xy(img):
    dx = deepcopy(img)
    dy = deepcopy(img)
    k = [1,-1]
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            dx[i][j] = abs(k[0]*img[i][j]+k[1]*img[i][j+1])
            dy[i][j] = abs(k[0]*img[i][j]+k[1]*img[i+1][j])
    return dx,dy


def gradient_magnitude(dx,dy):
    rho = deepcopy(dx)
    theta = np.array([[0.0 for y in range(len(dx[0]))] for x in range(len(dx))])
    for i in range(len(dx)):
        for j in range(len(dx[i])):
            rho[i][j] = math.sqrt(math.pow(dx[i][j],2)+math.pow(dy[i][j],2))
            theta[i][j] = math.atan2(dy[i][j],dx[i][j])
    return rho,theta


def non_max_suppress(rho,theta):
    for i in range(len(theta)):
        for j in range(len(theta[i])):
            theta[i][j] = round_angle(theta[i][j])



def print_matrix(matrix):
    for row in matrix:
        for pixel in row:
            print(pixel,end=' ')
        print('')


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
    rho,theta = gradient_magnitude(dx,dy)
    non_max_suppress(rho,theta)



    save_img('img/valve_magnitude.png',rho)
    #save_img('img/valve_theta.png',theta)


if __name__ == '__main__':
    driver()




# canny filter

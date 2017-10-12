# medium filter algorithm
from PIL import Image
import numpy as np
from copy import deepcopy
import sys


# functions for gaussian filter

def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels


def gen_2d(dimension):
    return [[0 for y in range(dimension)] for x in range(dimension)]


def get_attr(img,dim):
    M,N = len(img),len(img[0])
    L = 256
    A = dim*dim
    return M,N,L,A


# TODO fix this
# this offsets the image
def apply_filter(img,dim):
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


def driver():
    size = 3
    img = init_img('img/sample.jpg')
    fltrd = apply_filter(img,size)
    new_img = Image.fromarray(fltrd)
    new_img.save('img/sample_out.jpg')


if __name__ == '__main__':
    driver()


'''

# functions for gaussian filter

def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels


def gen_2d(dimension):
    return [[0 for y in range(dimension)] for x in range(dimension)]


def get_attr(img,dim):
    M,N = len(img),len(img[0])
    L = 256
    A = dim*dim
    return M,N,L,A


def apply_filter(img,dim):
    fltrd = deepcopy(img)
    off = int(dim/2)
    #w = [0 for x in range(dim*dim)]
    for i in range(off,len(img)-off):
        for j in range(off,len(img[i])-off):
            w_s = 0
            for s in range(i-off,i+off):
                for t in range(j-off,j+off):
                    #w[(s-i)*off+(t-j)] = img[s][t]
                    w_s += img[s][t]
            #fltrd[i][j] = median(w)
            fltrd[i][j] = w_s/(dim*dim)
    return fltrd


def driver():
    size = 3
    img = init_img('img/sample.jpg')
    fltrd = apply_filter(img,size)
    new_img = Image.fromarray(fltrd)
    new_img.save('img/sample_out.jpg')


if __name__ == '__main__':
    driver()

'''



'''

def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels


def median(a):
    a.sort()
    return a[(len(a)/2)+1]


def median_filter(img,dim):
    fltrd = deepcopy(img)
    off = int(dim/2)
    w = [0 for x in range(dim*dim)]
    for i in range(off,len(img)-off):
        for j in range(off,len(img[i])-off):
            for s in range(i-off,i+off):
                for t in range(j-off,j+off):
                    w[(s-i)*off+(t-j)] = img[s][t]
            fltrd[i][j] = median(w)
    return fltrd


def driver():
    size = 3
    img = init_img('img/sample.jpg')
    fltrd = median_filter(img,size)


if __name__ == '__main__':
    driver()

'''

# median filter

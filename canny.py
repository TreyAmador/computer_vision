# apply canny filter
from PIL import Image
import numpy as np
from copy import deepcopy
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
    fltrd = deepcopy(img)
    gaussian_smooth(img,fltrd,3)
    img_derivative(img)
    gradient_mag_ori(img)
    non_max_suppress(img)
    hysteresis_thresh(img)
    return fltrd


def gaussian_smooth(img,fltrd,dim):
    off = int(dim/2)
    for i in range(off,len(img)-off):
        for j in range(off,len(img[i])-off):
            fltr_sum = 0
            for s in range(i-off,i+off+1):
                for t in range(j-off,j+off+1):
                    fltr_sum += img[s][t]
            fltrd[i][j] = fltr_sum/(dim*dim)
    return fltrd


def img_derivative(img):
    pass


def gradient_mag_ori(img):
    pass


def non_max_suppress(img):
    pass


def hysteresis_thresh(img):
    pass



def driver():
    filepath = 'img/sample.jpg'
    img = init_img(filepath)
    fltrd = apply_canny(img)
    save_img('img/sample_out.jpg',fltrd)


if __name__ == '__main__':
    driver()




'''

# functions for gaussian filter

def gen_2d(dimension):
    return [[0 for y in range(dimension)] for x in range(dimension)]


def get_attr(img,dim):
    M,N = len(img),len(img[0])
    L = 256
    A = dim*dim
    return M,N,L,A


# good gaussian blur filter
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



# medium

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
    fltrd = gaussian_smooth(img,3)
    drvtv = derivative(fltrd)
    gradient_mag_ori(drvtv)
    non_max_suppress(img)
    hysteresis_thresh(img)
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


def gradient_mag_ori(img):
    magnitude = deepcopy(img)
    origin = deepcopy(img)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            pass
    return magnitude,origin


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


# medium

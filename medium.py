# medium filter algorithm
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
    for i in range(dim,len(img)-dim):
        for j in range(dim,len(img[i])-dim):
            fltr_sum = 0
            for s in range(dim):
                for t in range(dim):
                    fltr_sum += img[s+i][t+j]
            fltrd[i][j] = fltr_sum/(dim*dim)
    return fltrd


def driver():
    size = 3
    img = init_img('img/sample.jpg')
    #M,N,L,A = get_attr(img,size)
    fltrd = apply_filter(img,size)
    new_img = Image.fromarray(fltrd)
    new_img.save('img/sample_out.jpg')



if __name__ == '__main__':
    driver()





# medium

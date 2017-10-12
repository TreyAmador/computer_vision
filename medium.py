# medium filter algorithm
from PIL import Image
import numpy as np
import sys

'''
def init_img(filepath):
    def_img = filepath
    img = np.asarray(Image.open(def_img))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels
'''


def init_img(filepath):
    def_img = 'img/small.jpg'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = def_img
    img = np.asarray(Image.open(def_img))
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
    fltrd = gen_2d(len(img))
    off = int(dim/2)
    for i in range(off,len(img)-off):
        for j in range(off,len(img[i])-off):
            fltr_sum = 0
            print('(',i,'x',j,')',end=', ')
            for s in range(dim):
                for t in range(dim):
                    fltr_sum += img[s][t]
            fltrd[i][j] = fltr_sum/(dim*dim)
    return fltrd


def driver():
    size = 3
    img = init_img('img/small.jpg')
    print('image initialized')
    M,N,L,A = get_attr(img,3)
    print('attributes retrieved')
    filtered = apply_filter(img,M)
    print('image filtered')
    npimage = np.array(filtered)
    print('np array made')
    img_fltrd = Image.fromarray(npimage)
    img_fltrd.save('out.jpg')



if __name__ == '__main__':
    driver()


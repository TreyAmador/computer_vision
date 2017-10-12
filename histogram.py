# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with
#           sudo apt-get install imagemagick
from math import floor
from PIL import Image
import numpy as np
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


def img_dimensions(img):
    rows = len(img)
    cols = len(img[0])
    intensity = 256
    return rows,cols,intensity


def gen_intensity(img,M,N,L):
    intensity = np.zeros(L,np.uint8)
    for row in img:
        for pixel in row:
            intensity[pixel] += 1
    return intensity


def gen_transform(L,p_r):
    s = []
    for k in range(len(p_r)):
        s_k = 0
        for i in range(k+1):
            s_k += p_r[i]
        s.append(round((L-1)*s_k))
    return s


def gen_proportions(M,N,n_k):
    return np.array([n/(M*N) for n in n_k])


# perhaps this works?
def transform_image(img,tran_func):
    for i,pixel in enumerate(img):
        img[i] = img[tran_func[pixel]]
    return img


def trans_image(img,trans):
    for r,row in enumerate(img):
        for c,col in enumerate(row):
            img[r][c] = trans[col]


def driver():
    filepath = 'img/tab.jpg'
    img = init_img(filepath)
    M,N,L = img_dimensions(img)
    intensity = gen_intensity(img,M,N,L)
    p = gen_proportions(M,N,intensity)

    # gets weird around here
    s = gen_transform(L,p)

    trans_image(img,s)
    save_img('img/tab_out.jpg',img)


if __name__ == '__main__':
    driver()


# histogram equalization

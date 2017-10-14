
'''

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


def img_dimensions(img):
    rows = len(img)
    cols = len(img[0])
    intensity = 256
    return rows,cols,intensity


def gen_intensity(img,L):
    intensity = np.zeros(L,np.uint32)
    for row in img:
        for pixel in row:
            intensity[pixel] += 1
    return intensity


def get_max_min(p_r,L):
    mx,mn = -1,256
    for pixel in p_r:
        if pixel > mx:
            mx = pixel
        if pixel < mn:
            mn = pixel
    return mx,mn



def gen_cdf(p_r,L):
    s = [0 for x in p_r]
    for k in range(len(p_r)):
        s_k = 0
        for i in range(k+1):
            s_k += p_r[i]
        s[k] = round(s_k*(L-1))
    return s


def gen_transform(L,p_r,mx,mn):
    s = []
    for k in range(len(p_r)):
        s_k = 0
        for i in range(k+1):
            s_k += p_r[i]
        s.append(round((L-1)*s_k))
    return s


def transform_image(img,trans):
    for r,row in enumerate(img):
        for c,col in enumerate(row):
            img[r][c] = trans[col]


def print_img(img):
    for row in img:
        for pixel in row:
            print(pixel,end=' ')
        print('')


def save_img(filepath,pixels):
    img = Image.fromarray(pixels)
    img.save(filepath)


def driver():
    img = init_img('img/block.png')
    M,N,L = img_dimensions(img)
    p_r = gen_intensity(img,L)
    s = gen_cdf(p_r,L)
    transform_image(img,s)
    save_img('img/block_normalized.png',img)


if __name__ == '__main__':
    driver()

'''


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


def img_dimensions(img):
    rows = len(img)
    cols = len(img[0])
    intensity = 256
    return rows,cols,intensity


def gen_intensity(img,M,N,L):
    intensity = np.zeros(L,np.uint32)
    for row in img:
        for pixel in row:
            intensity[pixel] += 1
    return intensity


def gen_proportions(M,N,n_k):
    return np.array([n/(M*N) for n in n_k])


def get_max_min(img):
    max_i,min_i = -1,256
    for row in img:
        for pixel in row:
            if pixel > max_i:
                max_i = pixel
            if pixel < min_i:
                min_i = pixel
    return max_i,min_i


# input max, min
#   s_k * ( max - min ) + min
def gen_transform(L,p_r,mx,mn):
    s = []
    for k in range(len(p_r)):
        s_k = 0
        for i in range(k+1):
            s_k += p_r[i]
        s.append(round((L-1)*s_k))
    return s


def trans_image(img,trans):
    for r,row in enumerate(img):
        for c,col in enumerate(row):
            img[r][c] = trans[col]


def save_img(filepath,pixels):
    img = Image.fromarray(pixels)
    img.save(filepath)


def driver():
    filepath = 'img/bay.jpg'
    img = init_img(filepath)
    M,N,L = img_dimensions(img)
    intensity = gen_intensity(img,M,N,L)
    p = gen_proportions(M,N,intensity)
    mx,mn = get_max_min(img)

    # gets weird around here
    s = gen_transform(L,p,mx,mn)

    trans_image(img,s)
    save_img('img/bay_out.jpg',img)


if __name__ == '__main__':
    driver()


# histogram equalization

# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with
#           sudo apt-get install imagemagick
from math import floor
from PIL import Image
import numpy as np
import sys,cv2


def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.array(Image.open(filepath).convert('L'))
    return img


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


def gen_transform(L,p_r):
    s = []
    for k in range(len(p_r)):
        s_k = 0
        for i in range(k+1):
            s_k += p_r[i]
        s.append(round((L-1)*s_k))
    return s


# works against opencv histogram equalization
def trans_image(img,trans):
    for r,row in enumerate(img):
        for c,col in enumerate(row):
            img[r][c] = trans[col]


def histogram_cv2(img):
    return cv2.equalizeHist(img)


def save_img(filepath,pixels):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    t = filepath.split('.')
    filepath = '.'.join(t[:-1])+'_histogram.'+t[-1]
    img = Image.fromarray(pixels)
    img.save(filepath)


def driver():
    filepath = 'img/bay.jpg'
    img = init_img(filepath)
    M,N,L = img_dimensions(img)
    intensity = gen_intensity(img,M,N,L)
    p = gen_proportions(M,N,intensity)
    s = gen_transform(L,p)
    trans_image(img,s)
    save_img(filepath,img)

    hist_cv2_img = histogram_cv2(img)
    save_img('img/histogram_cv2.jpg',hist_cv2_img)


if __name__ == '__main__':
    driver()


# histogram equalization

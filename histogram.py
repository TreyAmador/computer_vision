# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with
#           sudo apt-get install imagemagick
import sys,cv2
import skvideo.io
import numpy as np
from PIL import Image
from math import floor


def init_img():
    def_img = 'img/forest.png'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = def_img
    img = cv2.imread(filepath)
    return img


def init_vid():
    def_vid = 'vid/aerial.mp4'
    skvideo.io.setFFmpegPath('/usr/bin/ffmpeg')
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = def_vid
    vidgen = skvideo.io.vreader(filepath)
    for index,frame in enumerate(vidgen):

        print(frame)


        # just for the purposes of testing one frame at a time
        if index >= 0:
            break


def hist_values(img):
    rows = len(img)
    cols = len(img[0])
    N = rows*cols
    intensity = 256


def init_hist(img):
    ''' assume each byte in pixel is same '''
    si = 255
    histogram = np.zeros(si,np.int8)
    hist_values(img)
    #print(histogram)

    #for elem in img:
    #    for pixel in elem:
    #        print(pixel[0],end=' ')
    #    print('')



def gen_transform(L,p_r):
    return np.array( \
        [round((L-1)*sum(p_r[i] for i in range(k+1))) \
        for k in range(len(p_r))])


def gen_proportions(L,M,N,n_k):
    return np.array([n/(M*N) for n in n_k])


# perhaps this works?
def transform_image(img,tran_func):
    for i,pixel in enumerate(img):
        img[i] = img[tran_func[pixel]]
    return img


def practice_driver():
    L = 8
    M = 64
    N = 64

    # number of pixels at certain intensity
    n_k = np.array([790,1023,850,656,329,245,122,81])

    p = gen_proportions(L,M,N,n_k)
    s = gen_transform(L,p)


    print(s)



def driver():
    img = init_img()
    init_hist(img)


if __name__ == '__main__':
    #driver()
    practice_driver()




# histogram equalization

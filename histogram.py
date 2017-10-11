# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with
#           sudo apt-get install imagemagick
import sys,cv2
import skvideo.io
import numpy as np
from PIL import Image


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





def practice_histogram(L,M,N,n_k):
    #p_r = np.array([n/(M*N) for n in n_k])
    p_r = np.array([0.19,0.25,0.21,0.16,0.08,0.06,0.03,0.02])
    for k,p in enumerate(p_r):
        s_k = 0
        for i in range(k+1):
            s_k += (L-1)*p_r[i]
        print(s_k)



def practice_driver():
    L = 8
    M = 64
    N = 64
    n_k = np.array([790,1023,850,656,329,245,122,81])
    practice_histogram(L,M,N,n_k)






def driver():
    img = init_img()
    #init_vid()
    init_hist(img)


if __name__ == '__main__':
    #driver()
    practice_driver()

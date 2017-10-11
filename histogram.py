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




def driver():
    img = init_img()
    #init_vid()
    init_hist(img)


if __name__ == '__main__':
    driver()

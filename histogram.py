# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with 
#           sudo apt-get install imagemagick
import sys,cv2
import numpy as np
from PIL import Image


def init_img():
    def_img = 'img/forest.pnm'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = def_img
    img = cv2.imread(filepath)
    return img


def init_hist(img):
    ''' assume each byte in pixel is same '''
    si = 255
    histogram = np.zeros(si,np.int8)
    #print(histogram)

    for elem in img:
        for pixel in elem:
            print(pixel[0],end=' ')
        print('')
    
    
    


def driver():
    img = init_img()
    init_hist(img)


if __name__ == '__main__':
    driver()



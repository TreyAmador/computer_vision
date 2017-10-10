# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with 
#           sudo apt-get install imagemagick
import sys,cv2,numpy
from PIL import Image


def init_img():
    def_img = 'img/sample.jpg'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = def_img
    img = cv2.imread(filepath)
    return img


def historgram(img):
    x = img[0]
    r = x[0]
    print(x)
    print(type(x))
    print('')
    print(r)
    print(type(r))
    #for x in img:
    #    print(x,end=' ')
    




def driver():
    img = init_img()
    historgram(img)


if __name__ == '__main__':
    driver()



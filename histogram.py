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
    #img = Image.open(filepath)
    #img.show()
    img = cv2.imread(filepath)
    print(type(img))
    





def historgram():
    pass
    #print(Image)



def driver():
    init_img()
    historgram()



if __name__ == '__main__':
    driver()



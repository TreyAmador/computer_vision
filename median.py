# medium filter algorithm
from PIL import Image
import numpy as np
from copy import deepcopy
import sys



def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels




def driver():
    img = init_img('img/sample.jpg')
    


if __name__ == '__main__':
    driver()


# median filter

from PIL import Image
import numpy as np
import cv2


def init_img(filepath):
    return Image.open(filepath)


def show_img(img):
    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def color_present(pixel_list,pixel):
    for rn,gn,bn in pixel_list:
        rm,gm,bm = pixel
        if rn == rm and gn == gm and bn == bm:
            return True
    return False


'''
def collect_colors(filepath):
    img = np.array(init_img(filepath))
    pixel_list = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            pixel = img[r,c]
            if not color_present(pixel_list,pixel):
                pixel_list.append(pixel)
    return pixel_list
'''


def collect_colors(filepath):
    img = np.array(init_img(filepath))
    pixel_list = list(set(tuple(v) for m2d in img for v in m2d))
    return pixel_list





def driver():
    pixel_list = collect_colors('bgimg/bg1.jpg')
    
    print()



if __name__ == '__main__':
    driver()


# end of file

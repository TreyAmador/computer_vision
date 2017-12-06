# removes background from an image


import numpy as np
import cv2



def out_path(filepath,suffix):
    fp = filepath.split('.')
    return fp[0]+suffix+'.'+fp[1]


def remove_foreground(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_green = np.array([50,150,50])
    upper_green = np.array([100,255,255])
    mask = cv2.inRange(hsv,lower_green,upper_green)
    res = cv2.bitwise_and(img,img,mask=mask)
    return res


def threshold(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,25,255,cv2.THRESH_BINARY_INV)
    return thresh


def background_removal(filepath):
    frgd = cv2.imread(filepath)
    mask = remove_foreground(frgd)
    thresh = threshold(mask)
    rest = img_restore(frgd,thresh)
    return thresh, rest


def img_restore(img,mask):
    gray = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return img*(gray/255)


def process_img(filepath):
    mask,rest = background_removal(filepath)
    cv2.imwrite(out_path(filepath,'_mask'),mask)
    cv2.imwrite(out_path(filepath,'_remove'),rest)


def driver():
    process_img('bgimg/img1_bg1.jpg')
    process_img('bgimg/img2_bg1.jpg')
    process_img('bgimg/img3_bg2.jpg')
    process_img('bgimg/img4_bg2.jpg')


if __name__ == '__main__':
    driver()









# end of file

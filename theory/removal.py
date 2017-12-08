from PIL import Image
import numpy as np
import cv2
import time



def init_img_PIL(filepath):
    return Image.open(filepath)


def init_img(filepath):
    return cv2.imread(filepath)


def save_img_PIL(filepath,img):
    img.save(filepath)


def show_img(img):
    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def color_present(pixel_list,pixel):
    rm,gm,bm = pixel
    for pix_elem in pixel_list:
        rn,gn,bn = pix_elem
        #print(rn,gn,bn,' x ',rm,gm,bm)
        if rn == rm and gn == gm and bn == bm:
            #print('match')
            return True
    return False


def collect_colors(filepath):
    img = np.array(init_img(filepath))
    pixel_list = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            pixel = img[r,c]
            if not color_present(pixel_list,pixel):
                pixel_list.append(pixel)
            print('collect',r,c)
    return pixel_list


'''
def collect_colors(filepath):
    img = init_img(filepath)
    pixel_list = list(set(tuple(v) for m2d in img for v in m2d))
    return pixel_list
'''


def remove_color_list(img,pixel_list):
    pixel_map = img.load()
    state_img = Image.new(img.mode,img.size)
    pixel_state = state_img.load()
    for i in range(state_img.size[1]):
        for j in range(state_img.size[0]):
            r,g,b = pixel_map[j,i]
            if color_present(pixel_list,(r,g,b)):
                pixel_state[j,i] = (0,0,0)
            else:
                pixel_state[j,i] = (r,g,b)
            print('remove',i,j)
    return state_img


def remove_background():
    img = init_img_PIL('bgimg/img1_bg1_scaled.jpg')
    pixel_list = collect_colors('bgimg/bg1_scaled.jpg')
    print('begin filtering')
    filtered_img = remove_color_list(img,pixel_list)
    save_img_PIL('bgimg/filtered_bg1_scaled.jpg',filtered_img)


def driver():
    remove_background()


if __name__ == '__main__':
    driver()


# end of file

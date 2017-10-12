# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with
#           sudo apt-get install imagemagick
from math import floor
from PIL import Image
import numpy as np
import sys


def init_img():
    def_img = 'img/tab.jpg'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = def_img
    img = np.asarray(Image.open(def_img))
    pixels = np.array([[col[0] for col in row] for row in img])
    return pixels


def img_dimensions(img):
    rows = len(img)
    cols = len(img[0])
    intensity = 256
    return rows,cols,intensity


'''
def gen_transform(L,p_r):
    return np.array( \
        [round((L-1)*sum(p_r[i] for i in range(k+1))) \
        for k in range(len(p_r))])
'''


def gen_intensity(img,M,N,L):
    intensity = np.zeros(L,np.uint8)
    for row in img:
        for pixel in row:
            intensity[pixel] += 1
    return intensity


def gen_transform(L,p_r):
    s = []
    for k in range(len(p_r)):
        s_k = 0
        for i in range(k+1):
            s_k += p_r[i]
        s.append(round((L-1)*s_k))
    return s



def gen_proportions(M,N,n_k):
    return np.array([n/(M*N) for n in n_k])


# perhaps this works?
def transform_image(img,tran_func):
    for i,pixel in enumerate(img):
        img[i] = img[tran_func[pixel]]
    return img


def trans_image(img,trans):
    for r,row in enumerate(img):
        for c,col in enumerate(row):
            img[r][c] = trans[col]
    return img


def driver():
    img = init_img()
    M,N,L = img_dimensions(img)
    intensity = gen_intensity(img,M,N,L)
    p = gen_proportions(M,N,intensity)

    # gets weird around here
    s = gen_transform(L,p)
    
    img = trans_image(img,s)
    new_img = Image.fromarray(img)
    new_img.save('img/output.jpg')




    '''
    L = 8
    M = 64
    N = 64

    # number of pixels at certain intensity
    n_k = np.array([790,1023,850,656,329,245,122,81])

    p = gen_proportions(M,N,n_k)
    s = gen_transform(L,p)
    print(s)
    '''



if __name__ == '__main__':
    driver()







'''
# additional functionality testing video
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

'''




# histogram equalization

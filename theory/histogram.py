# computer vision histogram filter
#   requires imagemagick to display pictures
#       install on linux with:
#           sudo apt-get install imagemagick


# import pathlib to test if file exists
from pathlib import Path
# image import and export module
from PIL import Image
# numpy for efficient arrays
import numpy as np
# import system for user arg handling
import sys
# import opencv for equalizeHist function
import cv2


def user_input():
    '''
        tests if user inputs filepath
        and parses it
     '''
     # tests if there are user arugments
    if len(sys.argv) > 1:
        # assign filepath
        filepath = sys.argv[1]
        # determines path of image
        img = Path(filepath)
        # tests if input image exists
        if img.is_file():
            # splits filepath by period
            t = filepath.split('.')
            # generates filename and extension
            name,ext = '.'.join(t[:-1]),'.'+t[-1]
            # return filename and extension
            return name,ext
        # condition if file does not exist
        else:
            # error message for user
            print('File',filepath,'not found')
            # program exits if file not found
            sys.exit(1)
    # condition if user does not input args
    else:
        # message for user if imporper number args
        print('Using default image')
        # returns default file if none found
        return 'img/bay','.jpg'


def init_img(filepath):
    '''
        initializes image
    '''
    # creates image from PIL module as np array
    # and converts it to black and white image
    return np.array(Image.open(filepath).convert('L'))


def save_img(filepath,pixels):
    '''
        outputs the image to same dir as input
        with additional info in file name
    '''
    # splits the image by period
    t = filepath.split('.')
    # appends file to contain algorithm of transformation
    filepath = '.'.join(t[:-1])+'_histogram.'+t[-1]
    # creates PIL image from numpy array
    img = Image.fromarray(pixels)
    # function to save image to directory
    img.save(filepath)


def img_dimensions(img):
    '''
        return image dimensions and byte size
    '''
    # returns size of image in rows, cols
    rows,cols = len(img),len(img[0])
    # intensity is number of possible values per byte
    intensity = 256
    # returns img size and highest possible intensity
    return rows,cols,intensity


def gen_intensity(img,M,N,L):
    '''
        generate intensity transform
    '''
    # creates an emtpy numpy array for intensity
    intensity = np.zeros(L,np.uint32)
    # iterates across all rows
    for row in img:
        # iterates across cols in row
        for pixel in row:
            # increments based on number of intensity value
            intensity[pixel] += 1
    # return intensity transform
    return intensity


def gen_proportions(M,N,n_k):
    '''
        normalizes the intensity value
        across the histogram transform array
        based on area of image
    '''
    # normalizes intensity transform
    # based on area of image
    return n_k/(M*N)


def gen_transform(L,p_r):
    '''
        iterates over normalized intensity transform
        summing number of pixels above certain intensity
        for use in transforming image pixels
    '''
    # generates emtpy 
    s = np.zeros(len(p_r),dtype=np.uint8)
    # iterates through each intensity value of histogram
    for k in range(len(p_r)):
        # a sum value above intensity value
        s_k = 0.0
        # iterate to sum above specified intensity value
        for i in range(k+1):
            # sum intensity above given values
            s_k += p_r[i]
        # normalize based on max intensity value
        s[k] = round((L-1)*s_k)
    # return transform
    return s


# works against opencv histogram equalization
def trans_image(img,trans):
    '''
        transform the image computed histogram
    '''
    # copies image to new result
    res = img.copy()
    # iterate over each row in image
    for r,row in enumerate(res):
        # iterate over col in each row
        for c,col in enumerate(row):
            # intensity at index of 
            # transform histogram
            # corresponds to intensity
            # at specific index in image
            res[r][c] = trans[col]
    # return transformed image
    return res


def histogram_equalization(img):
    M,N,L = img_dimensions(img)
    intensity = gen_intensity(img,M,N,L)
    p = gen_proportions(M,N,intensity)
    s = gen_transform(L,p)
    tran = trans_image(img,s)
    return tran
    

def histogram_opencv(img):
    '''
        uses opencv function to test
        histogram equalization
        to compare against
        newly implemented hist equ
        function
        NOTE: equalizeHist function was 
        used, since i get interpreter 
        error stating that cv2 does 
        not have 'histogram' function

    '''
    # returns histogram transform
    return cv2.equalizeHist(img)


def driver():
    '''
        main function calling the 
        hist equ algorithms of the program
        and saving the images
    '''
    # calls user input function for file paths
    path,ext = user_input()
    # initializes image from dir
    img = init_img(path+ext)
    # calls homemade hist equ function
    hist = histogram_equalization(img)
    # calls opencv hist equ function
    histcv = histogram_opencv(img)
    # saves image transformed by homemade hist
    save_img(path+ext,hist)
    # saves image transformed by opencv
    save_img(path+'_cv'+ext,histcv)


if __name__ == '__main__':
    '''
        entry point for program
    '''
    # calls main driver for program
    driver()


# histogram equalization

# medium filter algorithm

# import pathlib to test if file exists
from pathlib import Path
# PIL module to import and export images
from PIL import Image
# numpy to use efficienct arrays
import numpy as np
# sys to determine arg user input information
import sys
# opencv for the use of median filters
import cv2


def user_input():
    '''
        accepts user input for median filter
        including file names and size of filter
    '''
    # test if the proper number of inputs from user
    if (len(sys.argv)) > 2:
        # get file in directory
        filepath = sys.argv[1]
        # get size of median filter
        size_str = sys.argv[2]
        # get file in directory
        img = Path(filepath)
        # determines if file exists in directory
        if img.is_file():
            # splits filepath by period
            t = filepath.split('.')
            # generates filename and extension
            name,ext = '.'.join(t[:-1]),'.'+t[-1]
            # try expression to transform integer
            try:
                # transform size from string to int
                size = int(size_str)
            # if cannot be converted, raises ValueError
            except ValueError:
                # output message for imporper size input
                print('Second arg must be integer')
                # exit program if improper input
                sys.exit(1)
            # condition if try worked successfully
            else:
                # test if filter is odd
                if size % 2 == 0:
                    # if not odd, output to user
                    print('Filter size must be odd')
                    # exit program if not odd
                    sys.exit(1)
                # tests if size is too small
                if size < 3:
                    # output error message if filter too small
                    print('Filter must be 3 or greater')
                    # exit program if filter too small
                    sys.exit(1)
                # return filename and extension
                return name,ext,size
        # condition if file not found
        else:
            # tell user file not found
            print('File',filepath,'not found')
            # exit program gracefully if no file
            sys.exit(1)
    # condition if no user args
    else:
        # print to user that default image being used
        print('Using default image')
        # return default image filepath and args
        return 'img/valve','.png',3


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
    filepath = '.'.join(t[:-1])+'_median.'+t[-1]
    # creates PIL image from numpy array
    img = Image.fromarray(pixels)
    # function to save image to directory
    img.save(filepath)


def median(a,n):
    '''
        get median in series of numbers
    '''
    # sort numbers in incrementing order
    a.sort()
    # return median value of sorted array
    return a[n]


# good median blur
def apply_median_filter(img,dim):
    '''
        blurs the image using the median filter
    '''
    # gets offset of median filter
    off = int(dim/2)
    # makes copy of the image to be manipulated
    fltrd = img.copy()
    # generage the values for the median filter
    w = np.zeros(dim*dim,dtype=np.uint8)
    # generates value for middle element for filter
    n = int((dim*dim)/2+1)
    # iterate through each row of image
    # up through half of filter
    for i in range(off,len(img)-off):
        # iterate through column of each row
        for j in range(off,len(img[i])-off):
            # iterate through rows of filter
            for s in range(i-off,i+off+1):
                # iterate through column of each row
                for t in range(j-off,j+off+1):
                    # places each value of img into filter
                    w[(s-i+off)*dim+(t-j+off)] = img[s][t]
            # place median val of filter into 
            # designated index of filter
            fltrd[i][j] = median(w,n)
    # return img with median filter
    return fltrd


def blur_opencv(img,size):
    '''
        function to call default opencv blur
    '''
    # returns result of opencv blur function
    # of input filter size
    return cv2.blur(img,(size,size))


def gaussian_blur_opencv(img,size):
    '''
        function to call default opencv gaussian blur
    '''
    # returns result of opencv gaussian blur
    # the input size of filter given as tuple
    return cv2.GaussianBlur(img,(size,size),0)


def median_blur_opencv(img,size):
    '''
        function to call opencv median blur
    '''
    # returns result of median blur
    # based on input filter size
    return cv2.medianBlur(img,size)


def driver():
    '''
        driver calls the various assorted
        algorithms of the program
        and saves the images to dir
    '''
    # collect user input information
    name,ext,size = user_input()
    # initialize the image
    img = init_img(name+ext)
    # calls homemade median filter algorithm
    fltrd = apply_median_filter(img,size)
    # saves homemade median result image
    save_img(name+''+ext,fltrd)
    # calls opencv median blur
    blur_cv2 = blur_opencv(img,size)
    # saves opencv median blur
    save_img(name+'_blur_cv2'+ext,blur_cv2)
    # calls gaussian blur opencv function
    gauss_cv2 = gaussian_blur_opencv(img,size)
    # saves opencv gaussian blur function
    save_img(name+'_gaussian_blur_cv2'+ext,gauss_cv2)
    # calls median blur from opencv
    med_blur_cv2 = median_blur_opencv(img,size)
    # saves opencv median blur function
    save_img(name+'_median_blur_cv2'+ext,med_blur_cv2)

    
if __name__ == '__main__':
    '''
        entry point for the program
    '''
    # calls program driver
    driver()


# median filter

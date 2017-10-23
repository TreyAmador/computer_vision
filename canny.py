# canny edge detector for images

# import degrees and radian conversion functions
from math import radians,degrees
# import pathlib to test if file exists
from pathlib import Path
# import PIL for image import and export
from PIL import Image
# numpy for efficient array use
import numpy as np
# import system for user interfacing
import sys
# import opencv functionality
import cv2


def user_input():
    '''
        accepts user input for canny edge filter
        including file names and min and max threshold
    '''
    # test if the proper number of inputs from user
    if (len(sys.argv)) > 3:
        # get file in directory
        filepath = sys.argv[1]
        # get val of min threshold
        min_val_str = sys.argv[2]
        # get val of max threshold
        max_val_str = sys.argv[3]
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
                # transform min threshold from string to int
                min_val = int(min_val_str)
                # transform max threshold from string to int
                max_val = int(max_val_str)
            # if cannot be converted, raises ValueError
            except ValueError:
                # output message for imporper size input
                print('Second and third args must be integers')
                # exit program if improper input
                sys.exit(1)
            # condition if try worked successfully
            else:
                # testing if imporper min threshold
                if min_val < 0:
                    # assign default minimum value
                    min_val = 0
                # testing if imporper max threshold
                if max_val > 255:
                    # assign default max max val
                    max_val = 255
                # return filename and thresholds
                return name,ext,max_val,min_val
        # condition if file not found
        else:
            # tell user file not found
            print('File',filepath,'not found')
            # exit program gracefully if no file
            sys.exit(1)
    # condition if no user args
    else:
        # print to user that default image being used
        print('Using default image and threshold values')
        # return default image filepath and thresholds
        return 'img/valve','.png',200,100


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
    filepath = '.'.join(t[:-1])+'_canny.'+t[-1]
    # creates PIL image from numpy array
    img = Image.fromarray(pixels)
    # function to save image to directory
    img.save(filepath)


def new_pixels(img):
    '''
        generate new integer numpy array 
        of specified image dimensions
    '''
    # return new int numpy array with all zeros
    return np.zeros(shape=(len(img),len(img[0])),dtype=np.uint8)


def new_gradient(img):
    '''
        generate new float numpy array 
        of specified image dimensions
    '''
    # return new empty float numpy array
    return np.zeros(shape=(len(img),len(img[0])))


def new_tuple(img):
    '''
        generate new integer pair numpy array 
        of specified image dimensions
    '''
    # return new empty numpy array with pairs of integers
    return np.zeros(shape=(len(img),len(img[0])),dtype='2int8')


def init_kernels():
    '''
        return predefined Sobel kernels
    '''
    # generages the x-direction Sobel kernel
    kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    # generates the y-direction Sobel kernel
    ky = [[1,2,1],[0,0,0],[-1,-2,-1]]
    # returns the x and y kernel
    return kx,ky


def round_angle(rad):
    '''
        rounds the angle to 45 degree sectors
    '''
    # convert radians to degrees for easier use
    deg = degrees(rad)
    # returns indeces for angle between conditional degrees
    if -22.5 <= deg <= 22.5: return 0,1
    # returns indeces for angle between conditional degrees
    elif 22.5 < deg <= 67.5: return 1,1
    # returns indeces for angle between conditional degrees
    elif -67.5 <= deg < -22.5: return 1,-1
    # returns indeces for angle between conditional degrees
    elif 67.5 < deg <= 112.5: return 1,0
    # returns indeces for angle between conditional degrees
    elif -112.5 <= deg < -67.5: return 1,0
    # returns indeces for angle between conditional degrees
    elif 112.5 < deg <= 157.5: return 1,-1
    # returns indeces for angle between conditional degrees
    elif -157.5 <= deg < -112.5: return 1,0
    # returns indeces for angle between conditional degrees
    elif 157.5 < deg <= 180.0: return 0,1
    # returns indeces for angle between conditional degrees
    elif -180.0 <= deg < -157.5: return 1,1
    # returns a default index value
    else: return 0,1


def gaussian_blur(img):
    '''
        opencv gaussian blur
        step 1/5
    '''
    # default size of gaussian blur
    size = 3
    # returns the gaussian blur image
    return cv2.GaussianBlur(img,(size,size),0)


# not used in final program
def gaussian_smooth(img,dim):
    '''
        a homemade gaussian blur function
        the opencv function is used
        instead of this version
    '''
    # calculate offset of filter
    off = int(dim/2)
    # create new image to be filtered
    fltrd = new_pixels(img)
    # iterate through rows of image
    for i in range(off,len(img)-off):
        # iterate through columns in each row
        for j in range(off,len(img[i])-off):
            # a sum var for the filter
            fltr_sum = 0
            # iterate through the filter rows
            for s in range(i-off,i+off+1):
                # iterate through the filter columns per row
                for t in range(j-off,j+off+1):
                    # sum the elements at the filter
                    fltr_sum += img[s][t]
            # average filter sum, place at center of filter
            fltrd[i][j] = fltr_sum/(dim*dim)
    # return the filtered image
    return fltrd


def sobel_edge(img):
    '''
        Sobel edge filter
        creates x and y derivative
        for image and returns them
    '''
    # generate empty gradient for x direction
    dx = new_gradient(img)
    # generate empty gradient for y direction
    dy = new_gradient(img)
    # returns the gradients
    kx,ky = init_kernels()
    # iterates through each row of numpy image
    for i in range(1,len(img)-1):
        # iterates through each column of numpy row
        for j in range(1,len(img[i])-1):
            # indexes the rows of the kernels
            for m in range(len(kx)):
                # indexes the cols of the kernels
                for n in range(len(kx[m])):
                    # calculate sum of kernel and place
                    # in center of derivative x image
                    dx[i][j] += img[i+m-1][j+n-1] * kx[m][n]
                    # calculate sum of kernel and place
                    # in center of derivative x image
                    dy[i][j] += img[i+m-1][j+n-1] * ky[m][n]
    # returns the derivatives of the images
    return dx,dy


def gradient_magnitude(dx,dy):
    '''
        calculate the magnitude and angle
        of the dx and dy images
    '''
    # caclulate gradient by raising each element
    # to power of two, adding, and square rooting
    rho = np.power(np.power(dx,2.0)+np.power(dy,2.0),0.5)
    # calculate arctan of the dy and dx for each element
    theta = np.arctan2(dy,dx)
    # return gradient and angle of derivatives
    return rho,theta


def non_max_suppress(grad,theta):
    '''
        supress extra line thickness
    '''
    # copy gradient
    rho = grad.copy()
    # create matrix for rounded angle coordinates
    thetaXY = new_tuple(theta)
    # iterate though rows of angle matrix
    for i in range(len(theta)):
        # iterate through cols of each row of angle matrix
        for j in range(len(theta[i])):
            # round angle then return adjacent 
            # coordinates based on that angle
            thetaXY[i][j] = round_angle(theta[i][j])
    # iterate through rows gradient
    for r in range(1,len(grad)-1):
        # iterate through columns of rows of gradient
        for c in range(1,len(grad[r])-1):
            # return the coordinates of the angle
            i,j = thetaXY[r][c]
            # suppress pixel at i,j indeces if not greater than
            # adjacent pixels based on returned indeces
            if grad[r][c] <= grad[r+i][c-j] or grad[r][c] <= grad[r-i][c+j]:
                # suppress pixel at index if condition true
                rho[r][c] = 0
    # return suppressed magnitude gradient
    return rho


def max_3x3_2d(mtx,r,c):
    '''
        finds largest value in a 3x3 matrix
        of pixel values
    '''
    # sentinel value for current high
    high = -1
    # iterate through 3x3 patch of image
    # iterate through 3 rows
    for i in range(-1,2):
        # iterate through 3 columns of row
        for j in range(-1,2):
            # determines value is higher than current max
            if mtx[r+i][c+j] > high:
                # if condition is true, set highest to new highest
                high = mtx[r+i][c+j]
    # return highest value found in 3x3 matrix
    return high


def hysteresis(gradient,high=200,low=100):
    '''
        
    '''
    # creates matrix of values where 0 is 
    # below threshold and 1 is above threshold
    strong = (gradient > high)
    # creates edge 2d matrix where strong pixels are 255
    edges = np.array(strong,dtype=np.uint8) * 255
    # creates matrix where strong = 2, midrange = 1, weak = 0
    threshold = np.array(strong, dtype=np.uint8) + (gradient > low)
    # init list of strong pixels
    pixels = []
    # iterate through rows of threshold matrix
    for r in range(1, len(threshold)-1):
        # iterate through columns of each row of threshold matrix
        for c in range(1, len(threshold[r])-1):
            # tests if threshold pixel is midrange
            if threshold[r][c] == 1:
                # determine max of 3x3 matrix around r,c
                patch_max = max_3x3_2d(threshold,r,c)
                # determine if there is adjacent maximum matrix
                # and set current pixel to max value
                if patch_max == 2:
                    # add adjacent vals to pixel list
                    pixels.append((r, c))
                    # set edge to maximum value
                    edges[r][c] = 255
    # loop through strong pixels until there are none left
    while len(pixels) > 0:
        # create list of new pixels
        new_pixels = []
        # iterate through pixels
        for r,c in pixels:
            # query 8-adjacent pixels
            # test pixels above and below
            for s in range(-1, 2):
                # test pixels left and right
                for d in range(-1, 2):
                    # skip this if at r and c
                    # or center of 3x3 matrix
                    if s != 0 or d != 0:
                        # index for selected pixels
                        # offset with adjacent pixels
                        r2,c2 = r+s,c+d
                        # if midrange threshold and final matrix not white
                        if threshold[r2][c2] == 1 and edges[r2][c2] == 0:
                            # add this pixel of interest to list which 
                            # will allow checking adjacent pixels
                            new_pixels.append((r2, c2))
                            # make adjacent pixel strong/visible
                            edges[r2][c2] = 255
        
        # replace previous list of pixels with new list
        # to iterate adjacent pixels
        pixels = new_pixels
    # return the edges numpy array
    return edges


def canny_edge_detector(img,high,low):
    '''
        calls 5 functions to complete 
        canny edge detection
    '''
    # blur image with gauss filter
    blur = gaussian_blur(img)
    # calculate x and y derivatives with sobel filter
    dx,dy = sobel_edge(blur)
    # calculate gradient and angle matrices
    rho,theta = gradient_magnitude(dx,dy)
    # suppresses non-strongest pixels in region
    sup = non_max_suppress(rho,theta)
    # determine which midrange edges 
    # are adjacent to strong pixels 
    edges = hysteresis(sup,high,low)
    # return edge matrix
    return edges



def canny_opencv(img,high,low):
    '''
        function calls
        opencv canny edge detection
    '''
    # return opencv canny edge detection matrix
    return cv2.Canny(img,low,high)


def driver():
    '''
        program driver
        calling algorithm function
        and saving final images
    '''
    # user input returning filepath and threshold values
    name,ext,high,low = user_input()
    # create new image to be analyzed
    img = init_img(name+ext)
    # homemade canny edge detection algorithm with given thresholds
    canny_edge = canny_edge_detector(img,high,low)
    # opencv canny edge detection with same thresholds
    canny_edge_cv = canny_opencv(img,high,low)
    # saves the homemade canny filtered image
    save_img(name+ext,canny_edge)
    # saves the opencv canny filtered image
    save_img(name+'_cv'+ext,canny_edge_cv)


if __name__ == '__main__':
    '''
        entry point of progam
    '''
    # calls program driver
    driver()


# canny edge detection

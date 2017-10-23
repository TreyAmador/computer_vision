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
    '''
    size = 3
    return cv2.GaussianBlur(img,(size,size),0)


def sobel_edge(img):
    dx = new_gradient(img)
    dy = new_gradient(img)
    kx,ky = init_kernels()
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            for m in range(len(kx)):
                for n in range(len(kx[m])):
                    dx[i][j] += img[i+m-1][j+n-1] * kx[m][n]
                    dy[i][j] += img[i+m-1][j+n-1] * ky[m][n]
    return dx,dy


def gradient_magnitude(dx,dy):
	grad = np.power(np.power(dx,2.0)+np.power(dy,2.0),0.5)
	theta = np.arctan2(dy,dx)
	return grad,theta


def non_max_suppress(grad,theta):
	rho = grad.copy()
	thetaXY = new_tuple(theta)
	for i in range(len(theta)):
		for j in range(len(theta[i])):
			thetaXY[i][j] = round_angle(theta[i][j])
	for r in range(1,len(grad)-1):
		for c in range(1,len(grad[r])-1):
			i,j = thetaXY[r][c]
			if grad[r][c] <= grad[r+i][c-j] or grad[r][c] <= grad[r-i][c+j]:
				rho[r][c] = 0
	return rho


def max_3x3_2d(mtx,r,c):
	high = -1
	for i in range(-1,2):
		for j in range(-1,2):
			if mtx[r+i][c+j] > high:
				high = mtx[r+i][c+j]
	return high


def canny_edge_detector(gradient,high=200,low=50):
	strong = (gradient > high)
	edges = np.array(strong,dtype=np.uint8) * 255
	threshold = np.array(strong, dtype=np.uint8) + (gradient > low)
	pixels = []
	for r in range(1, len(threshold)-1):
		for c in range(1, len(threshold[r])-1):
			if threshold[r][c] == 1:
				patch_max = max_3x3_2d(threshold,r,c)
				if patch_max == 2:
					pixels.append((r, c))
					edges[r][c] = 255
	while len(pixels) > 0:
		new_pixels = []
		for r,c in pixels:
			for s in range(-1, 2):
				for d in range(-1, 2):
					if s != 0 or d != 0:
						r2,c2 = r+s,c+d
						if threshold[r2][c2] == 1 and edges[r2][c2] == 0:
							new_pixels.append((r2, c2))
							edges[r2][c2] = 255
		pixels = new_pixels
	return edges


if __name__ == '__main__':
    name,ext,high,low = user_input()
    img = init_img(name+ext)
    blur = gaussian_blur(img)
    dx,dy = sobel_edge(blur)
    grad,theta = gradient_magnitude(dx,dy)
    sup = non_max_suppress(grad,theta)
    edges = canny_edge_detector(sup,high,low)
    save_img('img/valve_final.png',edges)




# canny edge detection

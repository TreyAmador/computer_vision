from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.misc import imread, imshow
from PIL import Image
import numpy as np
from math import radians,degrees
import time
import math
import sys
import cv2


def init_img(filepath):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    img = np.asarray(Image.open(filepath).convert('L'))
    pixels = np.array(img)
    return pixels


def save_img(filepath,pixels):
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    t = filepath.split('.')
    filepath = '.'.join(t[:-1])+'_canny.'+t[-1]
    img = Image.fromarray(pixels)
    img.save(filepath)


def new_pixels(img):
    return np.zeros(shape=(len(img),len(img[0])),dtype=np.uint8)


def new_gradient(img):
    return np.zeros(shape=(len(img),len(img[0])))

def new_tuple(img):
	return np.zeros(shape=(len(img),len(img[0])),dtype='2int8')


def init_kernels():
	kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
	ky = [[1,2,1],[0,0,0],[-1,-2,-1]]
	return kx,ky


def round_angle(rad):
    deg = degrees(rad)
    if -22.5 <= deg <= 22.5:
        return 0,1
    elif 22.5 < deg <= 67.5:
        return 1,1
    elif -67.5 <= deg < -22.5:
        return 1,-1
    elif 67.5 < deg <= 112.5:
        return 1,0
    elif -112.5 <= deg < -67.5:
        return 1,0
    elif 112.5 < deg <= 157.5:
        return 1,-1
    elif -157.5 <= deg < -112.5:
        return 1,0
    elif 157.5 < deg <= 180.0:
        return 0,1
    elif -180.0 <= deg < -157.5:
        return 1,1
    else:
        return 0,1


def gaussian_blur(img):
	return cv2.GaussianBlur(img,(3,3),0)


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


def canny_edge_detector(gradient,high=91,low=31):
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
		for r, c in pixels:
			for dr in range(-1, 2):
				for dc in range(-1, 2):
					if dr == 0 and dc == 0:
						continue
					r2,c2 = r+dr,c+dc
					if threshold[r2][c2] == 1 and edges[r2, c2] == 0:
						new_pixels.append((r2, c2))
						edges[r2][c2] = 255
		pixels = new_pixels
	return edges


if __name__ == '__main__':
	img = init_img('img/valve.png')
	blur = gaussian_blur(img)
	dx,dy = sobel_edge(blur)
	grad,theta = gradient_magnitude(dx,dy)
	sup = non_max_suppress(grad,theta)
	edges = canny_edge_detector(sup)
	save_img('img/valve_final.png',edges)




# canny edge detection

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
	thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5
	thetaXY = new_tuple(theta)
	for i in range(len(theta)):
		for j in range(len(theta[i])):
			thetaXY[i][j] = round_angle(theta[i][j])
	return grad,theta,thetaQ,thetaXY


def non_max_suppress(im,grad,thetaQ,thetaXY):
	rho = grad.copy()
	for r in range(1,len(grad)-1):
		for c in range(1,len(grad[r])-1):
			i,j = thetaXY[r][c]
			if grad[r][c] <= grad[r+i][c-j] or grad[r][c] <= grad[r-i][c+j]:
				rho[r][c] = 0
	return rho


def canny_edge_detector(im,gradSup,theta,thetaQ, blur = 1, highThreshold = 91, lowThreshold = 31):
	strongEdges = (gradSup > highThreshold)
	thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (gradSup > lowThreshold)
	finalEdges = strongEdges.copy()
	currentPixels = []
	for r in range(1, im.shape[0]-1):
		for c in range(1, im.shape[1]-1):
			if thresholdedEdges[r, c] != 1:
				continue
			localPatch = thresholdedEdges[r-1:r+2,c-1:c+2]
			patchMax = localPatch.max()
			if patchMax == 2:
				currentPixels.append((r, c))
				finalEdges[r, c] = 1
	while len(currentPixels) > 0:
		newPix = []
		for r, c in currentPixels:
			for dr in range(-1, 2):
				for dc in range(-1, 2):
					if dr == 0 and dc == 0: continue
					r2 = r+dr
					c2 = c+dc
					if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
						newPix.append((r2, c2))
						finalEdges[r2, c2] = 1
		currentPixels = newPix
	return finalEdges


def convert_edges(edge):
	rho = np.array(edge,dtype=np.uint8)
	for i in range(len(rho)):
		for j in range(len(rho[i])):
			rho[i][j] *= 255
	return rho


if __name__ == '__main__':
	img = init_img('img/valve.png')
	img2 = gaussian_blur(img)
	dx,dy = sobel_edge(img2)
	grad,theta,thetaQ,thetaXY = gradient_magnitude(dx,dy)
	sup = non_max_suppress(img,grad,thetaQ,thetaXY)
	edges = canny_edge_detector(img,sup,theta,thetaQ)
	rho = convert_edges(edges)
	save_img('img/valve_final.png',rho)




# canny edge detection

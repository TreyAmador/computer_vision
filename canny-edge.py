import numpy as np
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


def gaussian_blur(img):
	gauss = np.array(img,dtype=np.float)
	return cv2.GaussianBlur(gauss,(3,3),0)


def sobel_edge(im2):
	im3h = convolve(im2,[[-1,0,1],[-2,0,2],[-1,0,1]])
	im3v = convolve(im2,[[1,2,1],[0,0,0],[-1,-2,-1]])
	return im3h,im3v


def gradient_magnitude(im3h,im3v):
	grad = np.power(np.power(im3h, 2.0) + np.power(im3v, 2.0), 0.5)
	theta = np.arctan2(im3v, im3h)
	thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5 #Quantize direction
	return grad,theta,thetaQ


def non_max_suppress(im,grad):
	gradSup = grad.copy()
	for r in range(im.shape[0]):
		for c in range(im.shape[1]):
			#Suppress pixels at the image edge
			if r == 0 or r == im.shape[0]-1 or c == 0 or c == im.shape[1] - 1:
				gradSup[r, c] = 0
				continue
			tq = thetaQ[r, c] % 4

			if tq == 0: #0 is E-W (horizontal)
				if grad[r, c] <= grad[r, c-1] or grad[r, c] <= grad[r, c+1]:
					gradSup[r, c] = 0
			if tq == 1: #1 is NE-SW
				if grad[r, c] <= grad[r-1, c+1] or grad[r, c] <= grad[r+1, c-1]:
					gradSup[r, c] = 0
			if tq == 2: #2 is N-S (vertical)
				if grad[r, c] <= grad[r-1, c] or grad[r, c] <= grad[r+1, c]:
					gradSup[r, c] = 0
			if tq == 3: #3 is NW-SE
				if grad[r, c] <= grad[r-1, c-1] or grad[r, c] <= grad[r+1, c+1]:
					gradSup[r, c] = 0
	return gradSup


def canny_edge_detector(im,gradSup,theta,thetaQ, blur = 1, highThreshold = 91, lowThreshold = 31):

	#Double threshold
	strongEdges = (gradSup > highThreshold)

	#Strong has value 2, weak has value 1
	thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (gradSup > lowThreshold)

	#Tracing edges with hysteresis
	#Find weak edge pixels near strong edge pixels
	finalEdges = strongEdges.copy()
	currentPixels = []
	for r in range(1, im.shape[0]-1):
		for c in range(1, im.shape[1]-1):
			if thresholdedEdges[r, c] != 1:
				continue #Not a weak pixel

			#Get 3x3 patch
			localPatch = thresholdedEdges[r-1:r+2,c-1:c+2]
			patchMax = localPatch.max()
			if patchMax == 2:
				currentPixels.append((r, c))
				finalEdges[r, c] = 1


	#Extend strong edges based on current pixels
	while len(currentPixels) > 0:
		newPix = []
		for r, c in currentPixels:
			for dr in range(-1, 2):
				for dc in range(-1, 2):
					if dr == 0 and dc == 0: continue
					r2 = r+dr
					c2 = c+dc
					if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
						#Copy this weak pixel to final result
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
	grad,theta,thetaQ = gradient_magnitude(dx,dy)
	sup = non_max_suppress(img,grad)
	edges = canny_edge_detector(img,sup,theta,thetaQ)
	rho = convert_edges(edges)
	save_img('img/valve_final.png',rho)








# canny edge detection

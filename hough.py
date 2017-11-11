import cv2
import numpy as np
from PIL import Image
import time


def canny_img(filepath):
	img = cv2.imread(filepath)
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(grey,30,100,apertureSize=3)
	return img,canny


def hough_lines(img,canny):
	lines = cv2.HoughLines(canny,1,np.pi/180.0,200)
	for line in lines:
		for rho,theta in line:
			a = np.cos(theta)
			b = np.sin(theta)
			x_0 = a*rho
			y_0 = b*rho
			x_1 = int(x_0+1000*(-b))
			y_1 = int(y_0+1000*(a))
			x_2 = int(x_0-1000*(-b))
			y_2 = int(y_0-1000*(a))
			cv2.line(img,(x_1,y_1),(x_2,y_2),(0,255,0),2)
	return img


def hough_lines_p(img,canny):
	minLineLength = 100
	maxLineGap = 10
	lines = cv2.HoughLinesP(canny,1,np.pi/180,100,minLineLength,maxLineGap)
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	return img


def driver():
	img,canny = canny_img('img/pool table.jpg')
	hough = hough_lines(img,canny)
	cv2.imwrite('img/canny_pool.jpg',canny)
	cv2.imwrite('img/hough_pool.jpg',hough)




if __name__ == '__main__':
	driver()


# end of file

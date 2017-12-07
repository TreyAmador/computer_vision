import numpy as np
import cv2


def init_img(filepath):
	return cv2.imread(filepath)


def show_img(img):
	img_min = cv2.resize(img,None,fx=0.5,fy=0.5)
	cv2.imshow('',img_min)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def filter_img(img):
	fltr = img.copy()
	lower_green = np.array([63,64,0])
	upper_green = np.array([255,255,33])
	mask = cv2.inRange(fltr,lower_green,upper_green)
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	dilation = cv2.dilate(opening,kernel,iterations = 1)
	res = cv2.bitwise_or(img,img,mask=dilation)
	return res


def hough_lines(img,mask):
	#img = cv2.imread('dave.jpg')
	#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = mask
	edges = cv2.Canny(gray,150,200,apertureSize = 3)

	show_img(edges)

	lines = cv2.HoughLines(edges,1,np.pi/90,110)

	for line in lines:
		for rho,theta in line:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1000*(-b))
		    y2 = int(y0 - 1000*(a))

		    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

	#cv2.imwrite('houghlines3.jpg',img)

	show_img(img)

	return img


def driver():
	img = init_img('img/pool table.jpg')
	fltr = filter_img(img)
	hough = hough_lines(img,fltr)


	#show_img(fltr)
	#cv2.imwrite('img/pool table filter.jpg',hough)


if __name__ == '__main__':
	driver()




# end of file

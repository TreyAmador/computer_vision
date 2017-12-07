import numpy as np
import cv2


def init_img(filepath):
	return cv2.imread(filepath)


def show_img(img):
	img_min = cv2.resize(img,None,fx=0.5,fy=0.5)
	cv2.imshow('',img_min)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def init_point(x,y):
	return { 'x':int(x), 'y':int(y) }


def init_line(x1,y1,x2,y2):
	return { 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2 }


def query_line(line):
    # return four points of lines from dict data struct
    return line['x1'],line['y1'],line['x2'],line['y2']


def intersection(line_a,line_b):
    '''
        calculate the intersection points of four lines
    '''
    # get four points from line a
    x1,y1,x2,y2 = query_line(line_a)
    # get four points from line b
    x3,y3,x4,y4 = query_line(line_b)
    # euclidean numerator
    pn = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    # return sentinel values if need to divide by zero
    if pn == 0:
        # return sentinels
        return -1,-1
    # calculate numerator for euclidean intersection
    px = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
    # calculate numerator for euclidean intersection
    py = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)
    # return intersection points between two lines
    return px/pn,py/pn


def get_intersections(lines,h,w):
	inter = []
	size = len(lines)
	for a in range(size-1):
		for b in range(1,size):
			x,y = intersection(lines[a],lines[b])
			p = init_point(x,y)
			not_present = True
			for i in inter:
				s,t = i['x'],i['y']
				if p['x'] == s and p['y'] == t:
					not_present = False
			if 0 <= y <= h and 0 <= x <= w and not_present:
				inter.append(p)
	min_index,min_value = min(
		enumerate([i['y'] for i in inter]),key=lambda p: p[1])
	del inter[min_index]
	return inter


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


def hough_lines(mask):
	#img = cv2.imread('dave.jpg')
	#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = mask
	edges = cv2.Canny(gray,150,200,apertureSize = 3)
	#show_img(edges)
	lines = cv2.HoughLines(edges,1,np.pi/90,110)
	hough_lines = []
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
			#cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
			hough_lines.append(init_line(x1,y1,x2,y2))
	#cv2.imwrite('houghlines3.jpg',img)
	#show_img(img)
	#return img
	return hough_lines


def driver():
	img = init_img('img/pool table.jpg')
	fltr = filter_img(img)
	lines = hough_lines(fltr)

	# put this elsewhere
	h,w,_ = img.shape
	inters = get_intersections(lines,h,w)

	for i in inters:
		cv2.circle(img,(i['x'],i['y']),20,(0,255,0))
	cv2.imwrite('img/pool hough circles.jpg',img)



	#show_img(fltr)
	#cv2.imwrite('img/pool table filter.jpg',hough)





if __name__ == '__main__':
	driver()




# end of file

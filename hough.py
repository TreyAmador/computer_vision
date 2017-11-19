import cv2
import numpy as np
from PIL import Image
import math


def init_img(filepath):
    return Image.open(filepath)


def init_img_brg(filepath):
    return cv2.imread(filepath)


def show_img(img):
    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_img(filepath,img):
	out = Image.fromarray(img)
	out.save(filepath)


def query_line(line):
    return line['x1'],line['y1'],line['x2'],line['y2']


def intersection(line_a,line_b):
    x1,y1,x2,y2 = query_line(line_a)
    x3,y3,x4,y4 = query_line(line_b)
    pn = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if pn == 0:
        return -1,-1
    px = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
    py = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)
    return px/pn,py/pn


def slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)


def img_filter(im):
	pixel_map = im.load()
	img = Image.new(im.mode,im.size)
	pixels_new = img.load()
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			r,g,b = pixel_map[i,j]
			if r < 26 and 63 < g and 60 < b:
				pixels_new[i,j] = (r,g,b)
			else:
				pixels_new[i,j] = (0,0,0)
	return np.array(img)


def canny_filter(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,150,200,apertureSize=3)
	return edges


def gen_line_collection(lines):
	hough_lines = []
	for line in lines:
		for rho,theta in line:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1600*(-b))
			y1 = int(y0 + 1600*(a))
			x2 = int(x0 + 1600*(b))
			y2 = int(y0 + 1600*(-a))
			hough_lines.append({
                'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                's':slope(x1,y1,x2,y2),
                'theta':math.atan2(y2-y1,x2-x1)
			})
	return hough_lines


def prune_hough_lines(hough_lines):
	prune_lines = []
	for i in range(len(hough_lines)-1):
		for j in range(i+1,len(hough_lines)):
			angle = abs(hough_lines[i]['theta']-hough_lines[j]['theta'])
			angle = angle*180/np.pi
			if angle < 2.0 and hough_lines[j] not in prune_lines:
				prune_lines.append(j)
	for i in prune_lines:
		del hough_lines[i]


def draw_lines(img,hough_lines):
	drawn = np.array(img)
	for line in hough_lines:
		x1,y1,x2,y2 = query_line(line)
		cv2.line(drawn,(x1,y1),(x2,y2),(0,255,0),3)
	return drawn


def clean_lines(hough_lines):
	pass


def detect_hough_lines(orig):
	im = orig.copy()
	img = img_filter(im)
	edges = canny_filter(img)
	lines = cv2.HoughLines(edges,1,np.pi/180,150)
	hough_lines = gen_line_collection(lines)
	prune_hough_lines(hough_lines)
	clean_lines(hough_lines)
	out = draw_lines(orig,hough_lines)
	return out


def detect_hough_circles(orig):
    img = cv2.cvtColor(np.array(orig),cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)
    cimg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                                param1=90,param2=20,minRadius=10,maxRadius=23)

    rad = 26
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0,:]
        for i in range(len(circles)):
            x,y = circles[i][0],circles[i][1]
            pu = img[y-rad][x]
            pd = img[y+rad][x]
            pr = img[y][x+rad]
            pl = img[y][x-rad]
            adj = [pu,pd,pr,pl]
            accum = 0
            for b,g,r in adj:
                delta = abs(int(g)-int(b))
                if delta > 10:
                    accum =+ 1
            if accum >= 1:
                circles[i] = (0,0,0)
        for i in circles:
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    out = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return out



def driver():
    img = init_img('img/pool table.jpg')
    hough_lines = detect_hough_lines(img)
    hough_circles = detect_hough_circles(img)
    write_img('img/pool_table_hough_lines.jpg',hough_lines)
    write_img('img/pool_table_hough_circle.jpg',hough_circles)



if __name__ == '__main__':
	driver()


# end of file

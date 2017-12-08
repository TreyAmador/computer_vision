import cv2
import numpy as np
from PIL import Image
import math
import sys


def rho_theta(x1,y1,x2,y2):
    d_x,d_y = x2-x1,y2-y1
    rho = math.sqrt(math.pow(d_x,2)+math.pow(d_y,2))
    theta = math.atan2(d_y,d_x)
    return rho,theta


def init_img(filepath):
    return Image.open(filepath)


def init_img_brg(filepath):
    return cv2.imread(filepath)


def show_img(img):
    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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



def detect_hough_lines():

    im = Image.open('img/pool table.jpg')
    pixelMap = im.load()
    img = Image.new(im.mode, im.size)
    pixelsNew = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            r,g,b = pixelMap[i,j]
            if r < 26 and 63 < g and 60 < b:
                pixelsNew[i,j] = (r,g,b)
            else:
                pixelsNew[i,j] = (0,0,0)

    img = np.array(img)
    h,w = len(img),len(img[0])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,150,200,apertureSize=3)

    # test for line intersection similarity of angle
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    hough_lines = []
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 + 1000*(b))
            y2 = int(y0 + 1000*(-a))
            hough_lines.append({
                'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                's':slope(x1,y1,x2,y2),
                'theta':math.atan2(y2-y1,x2-x1)
            })

    # prune lines
    prune_lines = []
    for i in range(len(hough_lines)-1):
        for j in range(i+1,len(hough_lines)):
            angle = abs(hough_lines[i]['theta']-hough_lines[j]['theta'])
            angle = angle*180/np.pi
            if angle < 2.0 and hough_lines[j] not in prune_lines:
                prune_lines.append(j)
    for i in prune_lines:
        del hough_lines[i]


    for line in hough_lines:
        x1,y1,x2,y2 = query_line(line)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


def detect_hough_circles():

    img = cv2.imread('img/pool table.jpg')
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

    show_img(img)


#detect_hough_lines()
detect_hough_circles()



def driver():
    pass


if __name__ == '__main__':
    driver()

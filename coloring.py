import cv2
import numpy as np
from PIL import Image
import math
from scipy.ndimage.morphology import binary_fill_holes


def rho_theta(x1,y1,x2,y2):
    d_x,d_y = x2-x1,y2-y1
    rho = math.sqrt(math.pow(d_x,2)+math.pow(d_y,2))
    theta = math.atan2(d_y,d_x)
    return rho,theta



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
            x1 = int(x0 + w*(-b))
            y1 = int(y0 + w*(a))
            x2 = int(x0 - w*(-b))
            y2 = int(y0 - w*(a))
            hough_lines.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2})

    for line in hough_lines:
        cv2.line(img,(line['x1'],line['y1']),(line['x2'],line['y2']),(0,255,0),3)

    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


def detect_hough_circles():
    img = cv2.imread('img/pool table.jpg',0)
    #img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)



    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=90,param2=20,minRadius=10,maxRadius=23)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)

        res = cv2.resize(cimg,None,fx=0.5,fy=0.5)
        cv2.imshow('',res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


detect_hough_circles()

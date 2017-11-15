import cv2
import numpy as np
from PIL import Image

im = Image.open('img/pool table.jpg')
pixelMap = im.load()
img = Image.new(im.mode, im.size)
pixelsNew = img.load()
for i in range(img.size[0]):
    for j in range(img.size[1]):
        r,g,b = pixelMap[i,j]
        if r < 26 and 60 < g and 60 < b:
            pixelsNew[i,j] = (r,g,b)
        else:
            pixelsNew[i,j] = (0,0,0)

img = np.array(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,150,200,apertureSize=3)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,100,6)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

res = cv2.resize(img,None,fx=0.5, fy=0.5)
cv2.imshow('',cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
cv2.waitKey(0)




'''

import cv2
import numpy as np
from PIL import Image

im = Image.open('img/pool table.jpg')
pixelMap = im.load()
img = Image.new(im.mode, im.size)
pixelsNew = img.load()
for i in range(img.size[0]):
    for j in range(img.size[1]):
        r,g,b = pixelMap[i,j]
        if r < 26 and 60 < g and 60 < b:
            pixelsNew[i,j] = (r,g,b)
        else:
            pixelsNew[i,j] = (0,0,0)

img = np.array(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200,apertureSize=3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,100,10)
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

res = cv2.resize(img,None,fx=0.5, fy=0.5)
cv2.imshow('',cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

'''

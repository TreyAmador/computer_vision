# import opencv module for computer vision modules
import cv2
# import numpy for image array
import numpy as np
# import image for use of pixel gradients
from PIL import Image
# math for trig functions
import math


def init_img(filepath):
    '''
        get image for use in pixel gradients
    '''
    return Image.open(filepath)


def init_img_brg(filepath):
    '''
        get image using opencv color arrangement
    '''
    return cv2.imread(filepath)


def show_img(img):
    '''
        display image for debugging purposes
    '''
    res = cv2.resize(img,None,fx=0.5, fy=0.5)
    cv2.imshow('',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_img(filepath,img):
    '''
        outputs image to directory
    '''
    # convert np array to PIL img
	out = Image.fromarray(img)
    # write the image data
	out.save(filepath)


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


def slope(x1,y1,x2,y2):
    # calcuate slope of two points
    return (y2-y1)/(x2-x1)


def img_filter(im):
    '''
        filter out unwanted pixel values
    '''
    # generate pixel map from original image`
	pixel_map = im.load()
    # create new image from original image
	img = Image.new(im.mode,im.size)
    # create new pixel map from new image
	pixels_new = img.load()
    # iterate over rows of pixels in image
	for i in range(img.size[0]):
        # iterate over cols of pixels in image
		for j in range(img.size[1]):
            # get single pixel from map
			r,g,b = pixel_map[i,j]
            # conditional if pixel values are to be filtered
			if r < 26 and 63 < g and 60 < b:
                # if true, return pixel of original image map
                # to new image map to be returned
				pixels_new[i,j] = (r,g,b)
            # condition if pixel value to be removed
			else:
                # if pixel is to be removed, set to black
				pixels_new[i,j] = (0,0,0)
    # return image as an np array
	return np.array(img)


def canny_filter(img):
    '''
        create img with canny lines
    '''
    # create grayscale version of same image
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # get edges for canny edge detector
	edges = cv2.Canny(gray,150,200,apertureSize=3)
    # return canny edge image
	return edges


def gen_line_collection(lines):
    '''
        generate collection of lines from hough transform
        to be used in printing to final output
    '''
    # init empty list of desired lines
	hough_lines = []
    # iterate past 0th element in hough line collection
	for line in lines:
        # iterate through each magnitude and angle
        # value in the collection of lists
		for rho,theta in line:
            # get x baseline
			a = np.cos(theta)
            # get y baseline
			b = np.sin(theta)
            # get init x point
			x0 = a*rho
            # get init y point
			y0 = b*rho
            # generate init point on x line
			x1 = int(x0 + 1600*(-b))
            # generate init point on y line
			y1 = int(y0 + 1600*(a))
            # generate final point on x line
			x2 = int(x0 + 1600*(b))
            # generate final point on y line
			y2 = int(y0 + 1600*(-a))
            # add line into dict of lines
			hough_lines.append({
                # create key for x and y vals
                'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                # create key for slope
                's':slope(x1,y1,x2,y2),
                # create key for angle of line
                'theta':math.atan2(y2-y1,x2-x1)
			})
    # return hough line dict
	return hough_lines


def prune_hough_lines(hough_lines):
    '''
        remove unecessary lines that should
        not be committed to final image
    '''
    # init list of final pruned lines list
	prune_lines = []
    # iterate over list of lines
	for i in range(len(hough_lines)-1):
        # iterate past current elem in lines
		for j in range(i+1,len(hough_lines)):
            # find angle difference
			angle = abs(hough_lines[i]['theta']-hough_lines[j]['theta'])
            # convert angle difference to degrees
			angle = angle*180/np.pi
            # condition to remove list if lines are too close together
            # skip if already present in list
			if angle < 2.0 and hough_lines[j] not in prune_lines:
                # remove line if not too close to other line
				prune_lines.append(j)
    # delete lines that are not necessary
	for i in prune_lines:
        # delete keyword for list
		del hough_lines[i]


def draw_lines(img,hough_lines):
    '''
        add the lines to the img
    '''
    # convert to np array from PIL img
	drawn = np.array(img)
    # iterate through hough lines
	for line in hough_lines:
        # get four points to represent lines
		x1,y1,x2,y2 = query_line(line)
        # draw line on the image
		cv2.line(drawn,(x1,y1),(x2,y2),(0,255,0),3)
    # return image with drawn-on lines
	return drawn


def clean_lines(hough_lines):
	pass


def detect_hough_lines(orig):
    '''
        function to draw hough lines onto image
    '''
    # copy the original image
	im = orig.copy()
    # remove pixels that interfere with hough line drawing
	img = img_filter(im)
    # generate canny edges image
	edges = canny_filter(img)
    # call opencv hough lines function
    # TODO elaborate on this, go through algorithm
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    # generate lines to place on image
	hough_lines = gen_line_collection(lines)
    # remove lines if angle too small
	prune_hough_lines(hough_lines)
    # clean lines to prevent overextension
	clean_lines(hough_lines)
    # draw lines onto final image
	out = draw_lines(orig,hough_lines)
    # return output image
	return out


def detect_hough_circles(orig):
    '''
        a function to draw circles around an image
    '''
    # convert color scheme from PIL to opencv version
    img = cv2.cvtColor(np.array(orig),cv2.COLOR_RGB2BGR)
    # convert color image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # blur the image with predefined median blur
    gray = cv2.medianBlur(gray,5)
    # convet from gray to opencv image scheme
    cimg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    # call hough circles opencv function
    # the values were determined empircally
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                                param1=90,param2=20,minRadius=10,maxRadius=23)

    # radian val greater than max radius of circle
    rad = 26
    # if circle list is empty, do not process further
    if circles is not None:
        # convert circles to 16bit circle list
        circles = np.uint16(np.around(circles))
        # get necessary elements from circle list
        circles = circles[0,:]
        # iterate through list of circles
        for i in range(len(circles)):
            # x and y points of circle
            x,y = circles[i][0],circles[i][1]
            # return color of pixel above circle
            pu = img[y-rad][x]
            # return color of pixel below circle
            pd = img[y+rad][x]
            # return color of pixel to right of circle
            pr = img[y][x+rad]
            # return color of pixel to left of circle
            pl = img[y][x-rad]
            # create list of all adjacent pixel colors
            adj = [pu,pd,pr,pl]
            # iterator for number of times pixel value is out of range
            accum = 0
            # iterate through pixels of adjacent colors
            for b,g,r in adj:
                # take difference between blue and green magnitude
                delta = abs(int(g)-int(b))
                # if mag is greater than 10
                # assume color is not pool table
                if delta > 10:
                    # penalized iterator increments
                    accum =+ 1
            # if too many bad values
            if accum >= 1:
                # clear the circle if it is not on the pool table
                circles[i] = (0,0,0)
        # iterate through found circles
        for i in circles:
            # add a green circle to the image
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # convert from opencv color scheme to PIL scheme
    out = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # return output image
    return out


def driver():
    '''
        calls the hough functions
    '''
    # initalize the pool table image
    img = init_img('img/pool table.jpg')
    # call main hough lines function
    hough_lines = detect_hough_lines(img)
    # call main hough circles function
    hough_circles = detect_hough_circles(img)
    # output the hough line image to dir
    write_img('img/pool_table_hough_lines.jpg',hough_lines)
    # output the hough circle image to dir
    write_img('img/pool_table_hough_circle.jpg',hough_circles)



if __name__ == '__main__':
    '''
        entry point of program
    '''
	driver()


# end of file

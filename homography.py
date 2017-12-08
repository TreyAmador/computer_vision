import numpy as np
import cv2


def init_img(filepath):
	return cv2.imread(filepath)


def show_img(img,fct=1):
	img_min = cv2.resize(img,None,fx=1/fct,fy=1/fct)
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


def swap_indeces(arr,a,b):
	arr[a],arr[b] = arr[b],arr[a]


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
	gray = mask
	edges = cv2.Canny(gray,150,200,apertureSize = 3)
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
			hough_lines.append(init_line(x1,y1,x2,y2))
	return hough_lines


def perspective_corners(img):
	h,w,_ = img.shape
	fltr = filter_img(img)
	lines = hough_lines(fltr)
	inters = get_intersections(lines,h,w)
	return inters


def overhead_corners(img):
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(grey,250,255,cv2.THRESH_BINARY)
	kernel = np.ones((7,7),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
	dilation = cv2.dilate(opening,kernel,iterations = 50)

	edges = cv2.Canny(dilation,150,200,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/90,200)
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
			hough_lines.append(init_line(x1,y1,x2,y2))

	lines_a = hough_lines[:-1]
	lines_b = hough_lines[1:]
	h,w,_ = img.shape
	inters = []
	for line_a in lines_a:
		for line_b in lines_b:
			x,y = intersection(line_a,line_b)
			p = init_point(x,y)
			not_present = True
			for i in inters:
				if i['x']-2 <= p['x'] <= i['x']+2 and i['y']-2 <= p['y'] <= i['y']+2:
					not_present = False
			if 0 <= p['x'] <= w and 0 <= p['y'] <= h and not_present:
				inters.append(p)
	return inters


def detect_hough_circles(img):
	'''
		a function to draw circles around an image
	'''
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
		circles = list(circles[0,:])
        # iterate through list of circles
		i = 0
		# continue to iterate until past length of array
		while i < len(circles):
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
				del circles[i]
			else:
				i += 1

		return circles


def homographic_transform(persp_inters,over_inters,vertices):
	persp = np.array([[i['x'],i['y'],1] for i in persp_inters])
	over = np.array([[i['x'],i['y'],1] for i in over_inters])
	H,status = cv2.findHomography(persp,over)
	trans = np.array([np.matmul(H, v) for v in vertices])
	projections = np.array([[t[0]/t[2],t[1]/t[2]] for t in trans],dtype=np.uint32)
	return projections


def project_images(img_vert,vertices,img_proj,projections):
	img = img_proj.copy()
	kernel = np.ones((5,5),np.uint8)
	lower_green = np.array([63,64,0])
	upper_green = np.array([255,255,33])
	fct = 25
	d_fct = fct*2
	for vert,proj in zip(vertices,projections):
		xv,yv = vert[0],vert[1]
		mat = np.zeros(shape=(fct*2,fct*2,3),dtype=np.uint8)
		cv2.circle(mat,(fct,fct),fct,(255,255,255),-1)
		roi = img_vert[yv-fct:yv+fct,xv-fct:xv+fct]
		circ = cv2.bitwise_and(roi,mat)
		smooth = 255-cv2.inRange(circ,lower_green,upper_green)
		clr_mask = cv2.cvtColor(smooth,cv2.COLOR_GRAY2BGR)
		circ_clr = cv2.morphologyEx(clr_mask, cv2.MORPH_OPEN, kernel)
		project = cv2.bitwise_and(circ,circ_clr)
		res_proj = cv2.resize(project,None,fx=2,fy=2)
		xp,yp = proj[0],proj[1]
		subsection = img[yp-d_fct:yp+d_fct,xp-d_fct:xp+d_fct]
		for i in range(res_proj.shape[0]):
			for j in range(res_proj.shape[1]):
				b,g,r = res_proj[i,j]
				if b != 0 and g != 0 and b != 0:
					subsection[i,j] = res_proj[i,j]
		img[yp-d_fct:yp+d_fct,xp-d_fct:xp+d_fct] = subsection
	return img


def driver():
	persp_img = init_img('img/pool table.jpg')
	persp_inters = perspective_corners(persp_img)
	over_img = init_img('img/pool overhead.jpg')
	over_inters = overhead_corners(over_img)

	swap_indeces(over_inters,0,3)

	circles = detect_hough_circles(persp_img)
	vertices = np.array([[c[0],c[1],1] for c in circles])
	projections = homographic_transform(persp_inters,over_inters,vertices)

	img = project_images(persp_img,vertices,over_img,projections)
	show_img(img,4)


if __name__ == '__main__':
	driver()




# end of file

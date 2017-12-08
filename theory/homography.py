# homographic projections from from one angle of pool table to another


# import numpy array for use of image manipulation
import numpy as np
# a number of utility functions for manipulating images
import cv2


def init_img(filepath):
	'''
		ease function to import images using cv2
	'''
	return cv2.imread(filepath)


def init_point(x,y):
	'''
		creates new dictionary representing
		an x and y point
	'''
	return { 'x':int(x), 'y':int(y) }


def init_line(x1,y1,x2,y2):
	'''
		creates new dictionary representing
		a line which the initial and final points
		as x and y coordinates
	'''
	return { 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2 }


def query_line(line):
    '''
		return four points of lines from dict data struct
	'''
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
	'''
		swaps the elements at specified indeces
		of a given array
	'''
	arr[a],arr[b] = arr[b],arr[a]


def get_intersections(lines,h,w):
	'''
		calculates the intesection point
		between two lines
	'''
	# init empty list of intersections
	inter = []
	# get the lize of the input lines
	size = len(lines)
	# dual loops to check each line against the other
	# to check the point of intersections
	# iterate through number of lines except last
	for a in range(size-1):
		# iterate through number of lines except first
		for b in range(1,size):
			# return the intersection point of any two lines
			x,y = intersection(lines[a],lines[b])
			# generate new point dict from given point coords
			p = init_point(x,y)
			# bool to represent if intersection point
			# is already present on list of intersections
			not_present = True
			# iterate through each intersection
			for i in inter:
				# get coords of intersection
				s,t = i['x'],i['y']
				# check if point is point in list
				if s-2 <= p['x'] <= s+2 and t-2 <= p['y'] <= t+2:
					# set bool to inter being present
					not_present = False
			# if intersection is in range of image
			if 0 <= y <= h and 0 <= x <= w and not_present:
				# add point to list of intersections
				inter.append(p)
	# return list of intersections
	return inter


def filter_img(img):
	# replicate original image
	fltr = img.copy()
	# get lower bounds of color for image
	lower_green = np.array([63,64,0])
	# get upper bounds of color for image
	upper_green = np.array([255,255,33])
	# get pixel map of specified range of colors
	mask = cv2.inRange(fltr,lower_green,upper_green)
	# generate new kernel to open image
	kernel = np.ones((3,3),np.uint8)
	# create image with combs over the holes in image
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	# inlarge the image for use in hough line transformation
	dilation = cv2.dilate(opening,kernel,iterations = 1)
	# or img with mask of dilation
	res = cv2.bitwise_or(img,img,mask=dilation)
	# return resulting image
	return res


def hough_lines(mask,thresh):
	'''
		get hough lines from given image
	'''
	# copy image to something that will be gray
	gray = mask.copy()
	# canny edge detection
	edges = cv2.Canny(gray,150,200,apertureSize = 3)
	# run hough line detection at given threshold
	lines = cv2.HoughLines(edges,1,np.pi/90,thresh)
	# init empty list of hough lines
	h_lines = []
	# iterate through found hough lines
	for line in lines:
		# iterate through rho and theta of given line
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
			x1 = int(x0 + 1000*(-b))
			# generate init point on y line
			y1 = int(y0 + 1000*(a))
			# generate final point on x line
			x2 = int(x0 - 1000*(-b))
			# generate final point on y line
			y2 = int(y0 - 1000*(a))
			# add line into dict of lines
			h_lines.append(init_line(x1,y1,x2,y2))
	# return list of hough lines
	return h_lines


def perspective_corners(img):
	'''
		find corners of obliquely angled table
	'''
	# get dimensions of image
	h,w,_ = img.shape
	# filter out all but the green color of pool table
	fltr = filter_img(img)
	# get the hough lines at given threshold
	lines = hough_lines(fltr,110)
	# calculate the intersections between given lines
	inters = get_intersections(lines,h,w)
	# get minimum index value
	min_index,min_value = min(
		enumerate([i['y'] for i in inters]),key=lambda p: p[1])
	# delete minimum intersection point
	del inters[min_index]
	# return the intersection list
	return inters


def overhead_corners(img):
	'''
		find corners of the pool table from overhead view
	'''
	# copy default image into grayscale for purposes of thresholding
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# divide color to white and black above certain threshold
	ret,thresh = cv2.threshold(grey,250,255,cv2.THRESH_BINARY)
	# generate kernal for image morph transformations
	kernel = np.ones((7,7),np.uint8)
	# clear the unecessary particles in the image
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
	# increase the size of the pool table
	dilation = cv2.dilate(opening,kernel,iterations = 50)
	# get list of hough lines
	h_lines = hough_lines(dilation,200)
	# get dimensions of shape
	h,w,_ = img.shape
	# get the intersection point between the lines
	inters = get_intersections(h_lines,h,w)
	# return the list of intersections
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
			# else iterate to next point
			else:
				# increment indeces
				i += 1
		# return list of circles
		return circles


def homographic_transform(persp_inters,over_inters,vertices):
	'''
		transform points from initial prespective to overhead
		transform the points, then return the transformed
		points by homography of the input vertices points
	'''
	# generate first perspective corner points
	persp = np.array([[i['x'],i['y'],1] for i in persp_inters])
	# generate points of other perspective overhead
	over = np.array([[i['x'],i['y'],1] for i in over_inters])
	# find homographic matrix from input points
	H,status = cv2.findHomography(persp,over)
	# generate transform points on list via matrix multiplication
	# of the found homographic 3x3 matrix
	trans = np.array([np.matmul(H, v) for v in vertices])
	# get list of projections transformed by homographic matrix
	projections = np.array([[t[0]/t[2],t[1]/t[2]] for t in trans],dtype=np.uint32)
	# return list of transformed points from matrix
	return projections


def mark_projections(orig,projections):
	'''
		mark x's where the points are projected
		from original image to the transform image
	'''
	# default size of projected image
	l = 20
	# make copy of original image
	img = orig.copy()
	# define the color red
	red = (0,0,255)
	# iterate through list of all projection points
	for proj in projections:
		# get x and y coords of each projection
		x,y = proj
		# get init and final points of lines
		# offset from the projection points
		x1,x2,y1,y2 = x-l,x+l,y-l,y+l
		# draw one stroke of projection 'x'
		cv2.line(img,(x1,y1),(x2,y2),red,3)
		# draw next stroke of projection 'x'
		cv2.line(img,(x1,y2),(x2,y1),red,3)
	# return image with projection x's drawn on it
	return img


def clean_projections(img_vert,vert,fct,lower,upper,kernel):
	'''
		clean images to be projected onto projection image
	'''
	# get points of original vertex images
	xv,yv = vert[0],vert[1]
	# generate empty array for circles to be drawn upon
	mat = np.zeros(shape=(fct*2,fct*2,3),dtype=np.uint8)
	# draw circle mask on the the small empty image
	cv2.circle(mat,(fct,fct),fct,(255,255,255),-1)
	# get region of interest based on vertex coords
	roi = img_vert[yv-fct:yv+fct,xv-fct:xv+fct]
	# get circle image of roi and material
	circ = cv2.bitwise_and(roi,mat)
	# smooth the background color of the circles
	smooth = 255-cv2.inRange(circ,lower,upper)
	# convert color from gray to bgr to get same num channels
	clr_mask = cv2.cvtColor(smooth,cv2.COLOR_GRAY2BGR)
	# clearn the unnecessary noise from the circle image
	circ_clr = cv2.morphologyEx(clr_mask, cv2.MORPH_OPEN, kernel)
	# and the clear circle from the cleaned projected circle
	project = cv2.bitwise_and(circ,circ_clr)
	# get result of image projection, increased by size
	res_proj = cv2.resize(project,None,fx=2,fy=2)
	# return result
	return res_proj


def project_images(img_vert,vertices,img_proj,projections):
	'''
		draw the projected circles onto the final image
	'''
	# copy the image
	img = img_proj.copy()
	# init a factor to account for offset of projected images
	fct = 25
	# double the size of project for size of increased projected images
	d_fct = fct*2
	# create lower bounds for image filter
	lower = np.array([63,64,0])
	# create upper bounds for image filter
	upper = np.array([255,255,33])
	# create kernel for image cleaning
	kernel = np.ones((5,5),np.uint8)
	# iterate through vertices images and projected images
	for vert,proj in zip(vertices,projections):
		# clean the image to be projected to make them more cohesive
		res_proj = clean_projections(img_vert,vert,fct,lower,upper,kernel)
		# get points of projected image
		xp,yp = proj[0],proj[1]
		# get subsection from original image
		subsection = img[yp-d_fct:yp+d_fct,xp-d_fct:xp+d_fct]
		# iterate through rows in projected image
		for i in range(res_proj.shape[0]):
			# iterate through columns in projected image
			for j in range(res_proj.shape[1]):
				# get color of pixel at i,j
				b,g,r = res_proj[i,j]
				# check if pixel is not black
				if b != 0 and g != 0 and b != 0:
					# if pixel is not black, replace with pixel
					# from the modified image
					# otherwise, ignore that pixel altogether
					subsection[i,j] = res_proj[i,j]
		# replace the section of the image with the modified subsection
		img[yp-d_fct:yp+d_fct,xp-d_fct:xp+d_fct] = subsection
	# return image with new projected sections
	return img


def driver():
	'''
		program driver
		organizes essential functions together
	'''
	# init image of the first pool table at angle
	persp_img = init_img('img/pool table.jpg')
	# get corners of that pool table
	persp_inters = perspective_corners(persp_img)
	# init image of overhead pool table
	over_img = init_img('img/pool overhead.jpg')
	# get corners of the overhead image
	over_inters = overhead_corners(over_img)
	# swap indeces so table at angle and
	# table overhead will be standardized
	swap_indeces(over_inters,0,3)
	# get hough circles representing balls
	circles = detect_hough_circles(persp_img)
	# get vertices of balls with 1
	# to allow for use of homographic transform
	# ie, matrix multiplication x vector
	vertices = np.array([[c[0],c[1],1] for c in circles])
	# apply homographic transformation to vertices
	projections = homographic_transform(persp_inters,over_inters,vertices)
	# mark x's on the image onto which the balls
	# are to be projected
	mark_img = mark_projections(over_img,projections)
	# write output image of x-marked image
	cv2.imwrite('img/pool table marked.jpg',mark_img)
	# generate image with pool balls projected onto it
	proj_img = project_images(persp_img,vertices,over_img,projections)
	# write image with the projection on the overhead pool table
	cv2.imwrite('img/pool table projections.jpg',proj_img)


if __name__ == '__main__':
	'''
		entry point of program
	'''
	driver()




# end of file

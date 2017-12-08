# image segmentation
# import opencv for k means image creation
import cv2
# import numpy for image array
import numpy as np
# matplotlib for visualization purposes
from matplotlib import pyplot as plt


def init_img(filepath):
    '''
        create bgr images from filepath
    '''
    # return bgr image
    return cv2.imread(filepath)


def init_img_gray(filepath):
    '''
        create grayscale image from filepath
    '''
    # return grayscale image
    return cv2.imread(filepath,0)


def convert_to_gray(img):
    '''
        convert bgr image to grayscale
    '''
    # return grayscale image
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def save_img(path,img):
    '''
        save image at specified filepath
    '''
    # save the image to specified filepath
    cv2.imwrite(path,img)


def show_img(img):
    '''
        display the image for purposes of debugging
    '''
    # show the image with nameless window
    cv2.imshow('',img)
    # wait until key is pressed so window
    # doesn't close prematurely
    cv2.waitKey()
    # destroy window once key is pressed
    cv2.destroyAllWindows()


def calc_intensity(gray):
    '''
        iterates through grayscale image and returns
        an array of the number of times the specific
        intensity of a pixel is found in the image
    '''
    # init an empty array of 256 elements
    gradient = [0 for x in range(256)]
    # iterate through the rows of the image
    for i in range(gray.shape[0]):
        # iterate through the cols of the image
        for j in range(gray.shape[1]):
            # increment element representing specified pixel value
            gradient[gray[i,j]] += 1
    # return gradient with counts
    return gradient


def k_means_image_intensity(gray):
    '''
        apply and return grayscale image
        representing k means intensity of an image
    '''
    # create a flat array representing gray image
    Z = gray.reshape((-1,1))
    # apply k means for input image
    res = apply_k_means_of_image(gray,Z)
    # return resulting k means transformed image
    return res


def k_means_image(img):
    '''
        apply and return color image
        representing k means clusters of an image
    '''
    # create a 3D array representing color image
    Z = img.reshape((-1,3))
    # apply k means for input image
    res = apply_k_means_of_image(img,Z)
    # return resulting k means transformed image
    return res


def apply_k_means_of_image(img,Z):
    '''
        apply the k means algorithm to various images
    '''
    # convert np array to floats of 4 bytes
    Z = np.float32(Z)
    # variables defining termination criteria for iterating algorithms
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # number of groups of k, the number of clusters\
    # that the data will be divided into
    K = 8
    # apply the k means algorithm
    # initialized by randomly choosing centroids
    # calculates the euclidean distance between the centroids
    # calcualte average of each cluster to create new centroid values
    # new centroids will be labeled with 0 or 1, depending on
    # whether or not the data is closer to C1 or C2, respectively
    # the centers is an array of the number of centers of clusters
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # create an array of 8 bit representing center of cluster values
    center = np.uint8(center)
    # create an array from labels the same size as final image
    res = center[label.flatten()]
    # reshape array of clusters into original image
    # using the shape specified by an image
    res2 = res.reshape((img.shape))
    # return the resulting gray k means intensity image
    return res2


def mean_shift_image(img):
    '''
        apply means shift to image based on
        opencv pyr mean shift algorithm
    '''
    # apply mean shift algorithm to shifted image
    # pyrMeanShiftFiltering applies the initial step
    # of meanshift segmentation of image
    # output is the posterized image with color gradients flattened
    # over a neighborhood of pixels for each pixel,
    # the average spatial value and average color
    # a gaussian pyramid of given levels is built
    shifted = cv2.pyrMeanShiftFiltering(img,10,10)
    # return shifted image to calling function
    return shifted


def driver():
    '''
        aggregates the k means and
        means shift algorithms
    '''
    # create new image from dir
    img = init_img('img/peppers.jpg')
    # apply k means algorithm to color image
    k_means = k_means_image(img)
    # generate gray image
    gray = convert_to_gray(img)
    # apply k means algorithm to gray image
    k_means_intensity = k_means_image_intensity(gray)
    # apply means shift algorithm to gray image
    shifted_img = mean_shift_image(img)
    # save the image to a dir at specified filepath
    save_img('img/peppers_k_means.jpg',k_means)
    # save the image to a dir at specified filepath
    save_img('img/peppers_k_means_intensity.jpg',k_means_intensity)
    # save the image to a dir at specified filepath
    save_img('img/peppers_mean_shifted.jpg',shifted_img)


if __name__ == '__main__':
    '''
        entry point of program
        calls program driver
    '''
    # calls driver for program
    driver()


# end of file

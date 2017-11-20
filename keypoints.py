'''
    The files requires the use of Python 2.7
    and OpenCV 2.4.11
'''
# import numpy for 2D image array
import numpy as np
# opencv module for sift, orb function
import cv2
# matplotlib allows data displaying
from matplotlib import pyplot as plt


def drawMatches(img1, kp1, img2, kp2, matches):
    '''
        concatinates two images together and add keypoints
        between the features of both images
        the keypoints also have cirlces at the ends of them
    '''
    # concatenates the two images together
    # init rows of first image
    rows1 = img1.shape[0]
    # init columns of first image
    cols1 = img1.shape[1]
    # init rows of second image
    rows2 = img2.shape[0]
    # init columns of second image
    cols2 = img2.shape[1]
    # create blank output image, init with zeros, makes a color image
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    # put image in new larger image container
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    # put next image to the right of the image container
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    # iterate through each points between both images
    for mat in matches:
        # get matching keypoints for 1st images
        img1_idx = mat.queryIdx
        # get matching keypoints for 2nd image
        img2_idx = mat.trainIdx
        # x - columns and y - rows
        (x1,y1) = kp1[img1_idx].pt
        # x - columns and y - rows
        (x2,y2) = kp2[img2_idx].pt
        # draws circles at x,y coordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        # draw circles shifted to the left
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        # draw line between the two points
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)
    # return the output image
    return out


def init_img_grey(filepath):
    '''
        initialize a greyscale image from dir
    '''
    # return greyscale image
    return cv2.imread(filepath,0)


def write_img(filepath,img):
    '''
        write the image to the specified filepath
    '''
    # opencv function to write to specified filepath
    cv2.imwrite(filepath,img)


def sift_keypoint_matching(img1,img2):
    # init the opencv sift detector
    sift = cv2.SIFT()
    # detect and compute function
    # generates list of keypoints
    kp1, des1 = sift.detectAndCompute(img1,None)
    # detect and compute function
    # generates list of keypoints
    kp2, des2 = sift.detectAndCompute(img2,None)
    # opencv has bfmatcher with default parameters
    bf = cv2.BFMatcher()
    # gen matches with bf matcher
    matches = bf.knnMatch(des1,des2, k=2)
    # init list of good matches
    good = []
    # iterate through each match
    for m,n in matches:
        # threshold of m and n distances by factor
        if m.distance < 0.75*n.distance:
            # append to list if passes threshold
            good.append(m)
    # draw matching lines on adjacent images
    img3 = drawMatches(img1,kp1,img2,kp2,good)
    # return matched images
    return img3


def orb_keypoint_matching(img1,img2):
    # initialize orb detector
    orb = cv2.ORB()
    # detect and compute function
    # generates list of keypoints
    kp1, des1 = orb.detectAndCompute(img1,None)
    # detect and compute function
    # generates list of keypoints
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create bfmatcher object with defined parameters
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # gen matches with bf matcher
    matches = bf.match(des1,des2)
    # the matches are sorted by their distance
    matches = sorted(matches, key = lambda x:x.distance)
    # draw list of first matches
    img3 = drawMatches(img1,kp1,img2,kp2,matches)
    # return adjacent images with matching lines
    return img3


def driver():
    '''
        driver function that calls different
        keypoint matching operators
    '''
    # initalize first image to match against other image
    img1 = init_img_grey('img/box.png')
    # initialize second image to match against other image
    img2 = init_img_grey('img/box_in_scene.png')
    # match lines using sift keypoints operator
    sift_img = sift_keypoint_matching(img1,img2)
    # match lines using orb keypoints operator
    orb_img = orb_keypoint_matching(img1,img2)
    # write sift matched lines to specified dir
    write_img('img/box_keypoint_sift.png',sift_img)
    # write orb matched lines to specified dir
    write_img('img/box_keypoint_orb.png',orb_img)


if __name__ == '__main__':
    '''
        entry point of the program
        calls driver
    '''
    driver()



# end of file

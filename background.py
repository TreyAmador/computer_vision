# removes background from an image


# import numpy array for use of image manipulation
import numpy as np
# a number of utility functions for manipulating images
import cv2


def out_path(filepath,suffix):
    '''
        outputs the image to the directory specified
        with an appended suffix to take advantage
        of the name of the original file
        and output with new suffix
    '''
    # split end dot
    fp = filepath.split('.')
    # return new filepath
    return fp[0]+suffix+'.'+fp[1]


def remove_foreground(img):
    '''
        remove the foreground from a mask image
    '''
    # convert bgr to hsv
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # lower bounds of background color
    lower_green = np.array([50,150,50])
    # upper bounds of background color
    upper_green = np.array([100,255,255])
    # get a mask with only green background preserved
    mask = cv2.inRange(hsv,lower_green,upper_green)
    # clear the unecessary particles in the image
    clr_mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    # bitwise and puts mask onto original image
    res = cv2.bitwise_and(img,img,mask=clr_mask)
    # return result of transformation
    return res


def threshold(img):
    '''
        makes mask binary black and white
    '''
    # convet image to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # create binary black and white image for purposes
    # of filtering out much of the noise
    ret,thresh = cv2.threshold(gray,25,255,cv2.THRESH_BINARY_INV)
    # return the thresholded image
    return thresh


def background_removal(filepath):
    '''
        remove the background green screen from image
    '''
    # import image at specified filepath
    frgd = cv2.imread(filepath)
    # remove the foreground to generate mask
    mask = remove_foreground(frgd)
    # make image binary black and white
    # for ease in processing
    thresh = threshold(mask)
    # put the pieces of the image together
    # based on black and white mask
    rest = img_restore(frgd,thresh)
    # return restored and thresholded images
    return thresh, rest


def img_restore(img,mask):
    '''
        restore colors from black and white mask
    '''
    # convert gray image to color image
    gray = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    # replace each white pixel with color from orig image
    return img*(gray/255)


def process_img(filepath):
    '''
        remove of each image
    '''
    # remove background of image
    # return mask and resulting image
    mask,rest = background_removal(filepath)
    # write mask to dir
    cv2.imwrite(out_path(filepath,'_mask'),mask)
    # write processed image to dir
    cv2.imwrite(out_path(filepath,'_remove'),rest)


def driver():
    '''
        calls the image processing function on each image
    '''
    # remove background and save image at path
    process_img('img/img1_bg1.jpg')
    # remove background and save image at path
    process_img('img/img2_bg1.jpg')
    # remove background and save image at path
    process_img('img/img3_bg2.jpg')
    # remove background and save image at path
    process_img('img/img4_bg2.jpg')


if __name__ == '__main__':
    '''
        entry point of the program
    '''
    driver()




# end of file

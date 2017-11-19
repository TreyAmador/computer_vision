# image blending
# import opencv for gaussian and laplacian pyramid function
import cv2
# import numpy for image array
import numpy as np
# import system for communication with dir
import sys


def init_img(filepath):
    '''
        import image with opencv functionality
    '''
    # return read image of specified filepath
    return cv2.imread(filepath)


def gen_gaussian(orig,depth):
    '''
        generate gaussian pyramids
    '''
    # copy original image
    img = orig.copy()
    # create list with inital image
    pyramid = [img]
    # iterate through list of specified depth
    for i in range(depth+1):
        # generate image from pyramid of lower res
        img = cv2.pyrDown(img)
        # add lower resolution image to pyramid
        pyramid.append(img)
    # return gaussian pyramid with smaller images
    return pyramid


def gen_laplacian(gauss,depth):
    '''
        create laplacian pyramid from gaussain
        of specified depth
    '''
    # create list with init element from gaussian pyramid
    pyramid = [gauss[depth]]
    # iterate through range backwards of specified depth
    for i in range(depth,0,-1):
        # create tuple of specified sizes
        size = (gauss[i-1].shape[1],gauss[i-1].shape[0])
        # travel up-pyramid
        gauss_up = cv2.pyrUp(gauss[i],dstsize=size)
        # subtract higher and lower sized images
        # to generate laplacian image
        laplace = cv2.subtract(gauss[i-1],gauss_up)
        # append laplacian image to laplacian pyramid
        pyramid.append(laplace)
    # return laplacian pyramid
    return pyramid


def blend_mask(laplace_a,laplace_b,gauss_mask):
    '''
        blend the laplacian images with the gaussian mask
    '''
    # init empty list of blended images
    laplace_sum = []
    # iterate through separate pyramids
    for l_a,l_b,g_m in zip(laplace_a,laplace_b,gauss_mask):
        # multiply gaussian by laplacian
        # divide by 255 to find proporation
        # of pixel relative to highest pixel value
        p_1 = (g_m/255)*l_b
        # multiply opposite of gaussian by laplacian
        p_2 = (1-(g_m/255))*l_a
        # append blend to sum pyramid
        laplace_sum.append(p_1+p_2)
    # return blended images
    return laplace_sum


def collapse(laplace_sum,depth):
    '''
        collapse image pyramid with opencv pyramid up
    '''
    # get first element of pyramid
    laplace_pyramid = laplace_sum[0]
    # last element fro pyramid
    last_pyr = laplace_sum[-1]
    # iterate through range of given depth
    for i in range(1,depth+1):
        # get tuple of given size from current collapsing sum
        size = (laplace_sum[i].shape[1],laplace_sum[i].shape[0])
        # set current collapsed pyramid to final image
        laplace_pyramid = cv2.pyrUp(laplace_pyramid,dstsize=size)
        # add current pyramid level to current summed final image
        laplace_pyramid = cv2.add(laplace_pyramid,laplace_sum[i])
    # return blended image
    return laplace_pyramid


def driver():
    '''
        driver of program that that calls gaussian/laplacian blending
    '''
    # init first image to be blended
    img_a = init_img('img/apple.jpg')
    # init second image to be blended
    img_b = init_img('img/orange.jpg')
    # init mask as image
    img_mask = init_img('img/mask512.jpg')
    # depth of pyramid
    depth = 5
    # create gassian pyramid from first image
    gauss_a = gen_gaussian(img_a,depth)
    # create gaussian pyramid from second image
    gauss_b = gen_gaussian(img_b,depth)
    # create gaussian pyramid for gaussian mask
    # remove unecessary elements
    gauss_mask = gen_gaussian(img_mask,depth)[-2::-1]
    # generate laplacian pyramid from gaussian
    laplace_a = gen_laplacian(gauss_a,depth)
    # generate laplacian pyramid from gaussian
    laplace_b = gen_laplacian(gauss_b,depth)
    # blend the pyramids
    blends = blend_mask(laplace_a,laplace_b,gauss_mask)
    # collapse pyramid to make final blended image
    blended_img = collapse(blends,depth)
    # write blended image to directory
    cv2.imwrite('img/apple_orange_blended.jpg',blended_img)


if __name__ == '__main__':
    '''
        entry point of the program
    '''
    driver()



# end of file

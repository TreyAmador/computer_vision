# image segmentation
import cv2
import numpy as np
from matplotlib import pyplot as plt


def init_img(filepath):
    return cv2.imread(filepath)


def init_img_gray(filepath):
    return cv2.imread(filepath,0)


def convert_to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def save_img(path,img):
    cv2.imwrite(path,img)


def show_img(img):
    cv2.imshow('',img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def calc_intensity(gray):
    gradient = [0 for x in range(256)]
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            gradient[gray[i,j]] += 1
    return gradient


# this needs updating
def k_means_intensity(gray):
    intensity = calc_intensity(gray)
    Z = np.float32(np.array(intensity))
    plt.hist(gray.ravel(),256,[0,256])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    K = 4
    compactness,labels,centers = cv2.kmeans(Z,K,None,criteria,10,flags)
    plt.hist(centers,5,[0,255],color='r')
    plt.show()


def k_means_image_intensity(gray):
    Z = gray.reshape((-1,1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((gray.shape))
    return res2


def k_means_image(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def driver():
    img = init_img('img/peppers.jpg')
    k_means = k_means_image(img)
    save_img('img/peppers_k_means.jpg',k_means)
    gray = convert_to_gray(img)
    k_means_intensity = k_means_image_intensity(gray)
    save_img('img/peppers_k_means_intensity.jpg',k_means_intensity)




if __name__ == '__main__':
    driver()




'''
import numpy as np
import cv2

img = cv2.imread('img/flower.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg

matplotlib.use('TkAgg')

def abs_sobel_thresh(img, orient='x',sobel_kernel=3 ,thresh=(0, 255)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if orient == 'x':
        _sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        _sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(_sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel>thresh[0])&(scaled_sobel<thresh[1])] = 1
    return s_binary

def mag_thresh(img,sobel_kernel=3 ,thresh_mag=(0,255)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    x_sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y_sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(x_sobel)
    abs_sobely = np.absolute(y_sobel)
    magnitude = np.sqrt(np.square(abs_sobelx)+np.square(abs_sobely))
    scaled_magnitude = np.uint8(255*magnitude/np.max(magnitude))
    s_binary = np.zeros_like(scaled_magnitude)
    s_binary[(scaled_magnitude>=thresh_mag[0])&(scaled_magnitude<=thresh_mag[1])] = 1
    return s_binary

def dir_thresh(img,sobel_kernel=3 ,thresh_=(0,np.pi/2)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    x_sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y_sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(x_sobel)
    abs_sobely = np.absolute(y_sobel)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    s_binary = np.zeros_like(direction)
    s_binary[(direction>=thresh_[0])&(direction<=thresh_[1])] = 1
    return s_binary

def binarization(img):
    gradx = abs_sobel_thresh(img, 'x', 7, (40, 150))
    grady = abs_sobel_thresh(img, 'y', 7, (40, 150))
    mag_binary = mag_thresh(img, 7, (40, 150))
    dir_binary = dir_thresh(img, 13, (0.8, 1.3))
    combined = np.zeros_like(img)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

# image = mpimg.imread('signs_vehicles_xygrad.png')
# gradx = abs_sobel_thresh(image, 'x', 7, (40, 150))
# plt.imshow(gradx)
# plt.show()
# grady = abs_sobel_thresh(image, 'y', 7, (40, 150))
# plt.imshow(grady)
# plt.show()
# mag_binary = mag_thresh(image, 7, (40, 150))
# plt.imshow(mag_binary)
# plt.show()
# dir_binary = dir_thresh(image, 13, (0.8, 1.3))
# plt.imshow(dir_binary)
# plt.show()

# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
# plt.imshow(combined)
# plt.show()

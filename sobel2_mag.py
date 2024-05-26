import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg

matplotlib.use('TkAgg')

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


image = mpimg.imread('signs_vehicles_xygrad.png')
sob = mag_thresh(image, 3, (30, 200))
# sob = cv.cvtColor(sob, cv.COLOR_BGR2GRAY)
print(sob.shape)
plt.imshow(sob)
plt.show()

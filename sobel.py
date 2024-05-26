import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg

matplotlib.use('TkAgg')

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if orient == 'x':
        _sobel = cv.Sobel(gray, cv.CV_64F, 1, 0)
    else:
        _sobel = cv.Sobel(gray, cv.CV_64F, 0, 1)
    abs_sobel = np.absolute(_sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel>thresh_min)&(scaled_sobel<thresh_max)] = 1
    return s_binary


image = mpimg.imread('signs_vehicles_xygrad.png')
sob = abs_sobel_thresh(image, 'x', 30, 200)
# sob = cv.cvtColor(sob, cv.COLOR_BGR2GRAY)
print(sob.shape)
plt.imshow(sob)
plt.show()

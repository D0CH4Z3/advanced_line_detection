import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg

matplotlib.use('TkAgg')

def hls_select(_image, thresh=(0.2, 0.8)):
    hls = cv.cvtColor(_image, cv.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    print(s_channel)
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel>thresh[0])&(s_channel<=thresh[1])] = 1
    return binary_output

image = mpimg.imread('test5.jpg')
# s_chan = hls_select(image, (10, 255))
hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
s_channel = hls[:, :, 2]
# cv.imshow('s', s_channel)
# cv.waitKey(0)
print(s_channel)
binary_output = np.zeros_like(s_channel)
binary_output[(s_channel > 30) & (s_channel <= 200)] = 1
print(binary_output)
cv.imshow('binary', binary_output)
cv.waitKey(0)
# plt.imshow(s_chan)
# plt.show()


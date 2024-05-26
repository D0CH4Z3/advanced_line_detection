import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def warp(img, src_points, desired_points):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    dst = np.float32(desired_points)
    trasnfer_matrice = cv2.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, trasnfer_matrice, img_size, flags=cv.INTER_LINEAR)
    return warped


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if orient == 'x':
        _sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        _sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(_sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return s_binary


def mag_thresh(img, sobel_kernel=3, thresh_mag=(0, 255)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    x_sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y_sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(x_sobel)
    abs_sobely = np.absolute(y_sobel)
    magnitude = np.sqrt(np.square(abs_sobelx) + np.square(abs_sobely))
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    s_binary = np.zeros_like(scaled_magnitude)
    s_binary[(scaled_magnitude >= thresh_mag[0]) & (scaled_magnitude <= thresh_mag[1])] = 1
    return s_binary


def dir_thresh(img, sobel_kernel=3, thresh_=(0, np.pi / 2)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    x_sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y_sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(x_sobel)
    abs_sobely = np.absolute(y_sobel)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    s_binary = np.zeros_like(direction)
    s_binary[(direction >= thresh_[0]) & (direction <= thresh_[1])] = 1
    return s_binary


def hls_select(_image, thresh=(0, 255)):
    hls = cv.cvtColor(_image, cv.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def binarization(img):
    gradx = abs_sobel_thresh(img, 'x', 5, (40, 100))
    grady = abs_sobel_thresh(img, 'y', 5, (40, 100))
    mag_binary = mag_thresh(img, 5, (40, 100))
    dir_binary = dir_thresh(img, 13, (0.7, 1.3))
    HLS = hls_select(img, (110, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | HLS == 1] = 1
    return combined

# def HLS_fix(img, thresh):
#     dir_binary = dir_thresh(img, 13, (0.7, 1.3))
#     HLS = hls_select(img, thresh)
#     combined = np.zeros_like(dir_binary)
#     combined[HLS == 1] = 1
#     return combined

# image = cv.imread(r'test5.jpg')

# binarized = binarization(img)
# plt.imshow(binarized)
# plt.show()


# plt.imshow(img)
# plt.plot(735, 475, '.')  # top right
# plt.plot(1120, 720, '.')  # bottom right
# plt.plot(180, 720, '.')  # bottom left
# plt.plot(555, 475, '.')  # top left
# plt.show()
#
# source_points = [[735, 475], [1120, 720], [180, 720], [555, 475]]
# destination_points = [[1000, 0], [1000, 720], [250, 720], [250, 0]]

# final = warp(image, source_points, destination_points)
# cv.imshow('hello', final)
# cv.waitKey(0)

# ready = binarization(image)
# final = warp(ready, source_points, destination_points)
# cv.imshow('hello', final)
# cv.waitKey(0)

# HL_S = HLS_fix(image, (90, 255))
# cv.imshow('l', HL_S)
# cv.waitKey(0)
# plt.imshow(binarization(image))
# plt.show()

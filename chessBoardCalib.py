import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

matplotlib.use('TkAgg')

# image = cv.imread(r'./test1.jpg')
# cv.imshow('image', image)
# cv.waitKey(0)

def calibration_coeffs(img):
    # objpoints = []
    # imgpoints = []

    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        # imgpoints.append(corners)
        # objpoints.append(objp)
        img = cv.drawChessboardCorners(img, (9,6), corners, ret)
        cv.imshow("image", img)
        cv.waitKey(0)
        return corners, objp

def calibrating_camera(obj_points, img_points, img):
    _gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape = _gray.shape[::-1]
    ret, mtx_, dist_, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, shape_, None, None)
    dst_image = cv.undistort(img, mtx_, dist_, none, mtx_)
    return mtx_, dist_

# calibration_coeffs(image)

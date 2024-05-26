# importing libraries
import cv2 as cv
import numpy as np
import glob


def calibration_coeffs(img):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        return corners, objp


def calibrating_camera(obj_points_, img_points_, img):
    _gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape_ = _gray.shape[::-1]
    ret, mtx_, dist_, rvecs, tvecs = cv.calibrateCamera(obj_points_, img_points_, shape_, None, None)
    dst_image = cv.undistort(img, mtx_, dist_, None, mtx_)
    return dst_image


def warp(img, src_points, desired_points):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    dst = np.float32(desired_points)
    trasnfer_matrice = cv.getPerspectiveTransform(src, dst)
    un_transfer_matrice = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(img, trasnfer_matrice, img_size, flags=cv.INTER_LINEAR)
    return warped, un_transfer_matrice


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
    gradx = abs_sobel_thresh(img, 'x', 3, (20, 100))
    grady = abs_sobel_thresh(img, 'y', 3, (20, 100))
    mag_binary = mag_thresh(img, 9, (30, 100))
    dir_binary = dir_thresh(img, 13, (0.7, 1.3))
    HLS = hls_select(img, (180, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | HLS == 1] = 1
    return combined


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # print("+++++++++++++++++++++++++++++++++++++++")
    # print(left_fit)
    # print(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return ploty, left_fit, right_fit, left_fitx, right_fitx


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fit, right_fit, ploty, left_fitx, right_fitx


def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit_, right_fit_, ploty, left_fitx, right_fitx = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    return ploty, left_fit_, right_fit_, left_fitx, right_fitx


def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad

import numpy as np
from utilities import *

images = glob.glob(r'./camera_cal/calibration*.jpg')


# road width = 2.35m

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


source_points = [[735, 475], [1120, 720], [180, 720], [555, 475]]
destination_points = [[1000, 0], [1000, 720], [250, 720], [250, 0]]
objpoints = []
imgpoints = []
right_line = Line()
left_line = Line()

for fname in images:
    img = cv.imread(fname)
    corners, objp = calibration_coeffs(img)
    imgpoints.append(corners)
    objpoints.append(objp)

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('project_video.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

    # Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        image = calibrating_camera(objpoints, imgpoints, frame)
        warped_image, Minv = warp(image, source_points, destination_points)
        binarized_image = binarization(warped_image)

        if right_line.detected and left_line.detected:
            plt_y, l_fit, r_fit, left_fitx, right_fitx = search_around_poly(
                binarized_image, right_line.current_fit,
                left_line.current_fit)
        else:
            plt_y, l_fit, r_fit, left_fitx, right_fitx = fit_polynomial(binarized_image)
        l_curve, r_curve = measure_curvature_real(plt_y, l_fit, r_fit)
        right_line.detected = True
        left_line.detected = True

        left_line.current_fit = l_fit
        right_line.current_fit = r_fit

        left_line.recent_xfitted.append(np.mean(left_fitx))
        right_line.recent_xfitted.append(np.mean(right_fitx))

        left_line.bestx = np.mean(left_line.recent_xfitted)
        right_line.bestx = np.mean(right_line.recent_xfitted)

        left_line.radius_of_curvature = l_curve
        right_line.radius_of_curvature = r_curve

        left_line.diffs = left_line.diffs - l_fit
        right_line.diffs = right_line.diffs - r_fit

        left_line.allx = left_fitx
        right_line.allx = right_fitx

        left_line.ally = plt_y
        right_line.ally = plt_y

        left_line.line_base_pos = 1.175 - (left_line.recent_xfitted[-1] - right_line.recent_xfitted[-1]) * (
                    3.7 / 700) / 2
        right_line.line_base_pos = 1.175 - (right_line.recent_xfitted[-1] - left_line.recent_xfitted[-1]) * (
                    3.7 / 700) / 2

        # print(left_line.line_base_pos)

        binarized_image_zero = np.zeros_like(binarized_image).astype(np.uint8)
        color_binarized_image = np.dstack((binarized_image_zero, binarized_image_zero, binarized_image_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, plt_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plt_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv.fillPoly(color_binarized_image, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newbinarized_image = cv.warpPerspective(color_binarized_image, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv.addWeighted(frame, 1, newbinarized_image, 0.3, 0)
        cv.imshow('result', result)

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()

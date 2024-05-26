#########################################################################
from change_prespective import*

image = cv.imread(r'test5.jpg')

source_points = [[735, 475], [1120, 720], [180, 720], [555, 475]]
destination_points = [[1000, 0], [1000, 720], [250, 720], [250, 0]]

binarized_image = binarization(image)
warped_binary = warp(binarized_image, source_points, destination_points)
# cv.imshow('warped binary image', warped_binary)
# cv.waitKey(0)
#########################################################################

histogram = np.sum(warped_binary[warped_binary.shape[0]//2:,:], axis=0)
# plt.plot(histogram)
# plt.show()
out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
midpoint = int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# cv.imshow('out image', out_img)
# cv.waitKey(0)

# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

# Set height of windows - based on nwindows above and image shape
window_height = int(binary_warped.shape[0]//nwindows)
# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated later for each window in nwindows
leftx_current = leftx_base
rightx_current = rightx_base

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

from chessBoardCalib import *
import glob

images = glob.glob(r'./camera_cal/calibration*.jpg')

objpoints = []
imgpoints = []

for fname in images:
    img = cv.imread(fname)
    corners, objp = calibration_coeffs(img)
    imgpoints.append(corners)
    objpoints.append(objp)

print("this are obj points")
print(objpoints)
print("=========================")
print("this are image points")
print(imgpoints)

# img = cv.imread(r'./camera_cal/calibration20.jpg')
# cv.imshow('image', img)
# cv.waitKey(0)
#
# objpoints = []
# imgpoints = []
#
# objp = np.zeros((6 * 9, 3), np.float32)
# objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
#
# if ret == True:
#     imgpoints.append(corners)
#     objpoints.append(objp)
#     img = cv.drawChessboardCorners(img, (9, 6), corners, ret)
#     cv.imshow("image", img)
#     cv.waitKey(0)

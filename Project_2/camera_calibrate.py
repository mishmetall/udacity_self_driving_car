import cv2
import glob
import pickle
import numpy as np
from os.path import join, exists

def calibrate_camera(path_to_calib_images, plot=False):

    calib_file = join(path_to_calib_images, "calibration.pkl")
    if exists(calib_file):
        with open(calib_file, "rb") as f:
            data = pickle.load(f)

        return data["dist"], data["mtx"]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    calib_images = glob.glob(join(path_to_calib_images, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    imshape = None
    for fname in calib_images:
        img = cv2.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imshape = gray.shape

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if plot:
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(0)

    if plot:
        cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)

    # Save calibration once its made
    with open(calib_file, 'wb') as f:
        pickle.dump({"dist": dist, "mtx": mtx}, f)

    return dist, mtx

#!/usr/bin/env python
# author: Xiao Wang
# camera calibration class
# --------------------------------


import pickle
from glob import glob
import cv2
import numpy as np
import os

class Calibrator(object):
    def __init__(self, path, shape, load):
        # load calibration matrix
        if load:
            with open("calibration.dat", "rb") as f:
                self.calib = pickle.load(f)
                print("Calibration load successfully!")
        # calculate calibration matrix
        else:
            self.calib = get_calibration(path, shape)
            print("Calibration calculate and write successfully!")
            with open("calibration.dat", 'wb') as f:
                pickle.dump(self.calib, file=f)
    # undistorted images
    def undistorted(self, image):
        """ undistorted image using calculated calibration matrix
        @param:
            image: input colored images 
        @output:
            dst: undistorted colored image    
        """
        dst= cv2.undistort(image, self.calib["mtx"], self.calib["dist"], None, self.calib["mtx"])
        return dst

# calculate the calibration of the camera
def get_calibration(path, shape):
    """ using all test images to generate caibration matrix
        @param:
            path: relative path to calibration images
            shape: tuple of two, the number of crossover in the image
        @output:
            calibration: a dictionary contains all information needed for calibration
    """
    obj_points = []
    img_points = []
    files = os.listdir(path)
    success_count = 0
    
    objp = np.zeros((shape[0]*shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
    # read all images for calculation
    for file in files:
        img = cv2.imread(path+file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, shape ,None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            success_count += 1
    print("Total image processed(contains chessboard) {}".format(success_count))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[0:2], None, None)
    calibration = {"objpoints": obj_points, "imgpoints": img_points, "mtx":mtx, "dist":dist}
    return calibration
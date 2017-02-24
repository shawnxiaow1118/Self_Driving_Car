#!/usr/bin/env python
# author: Xiao Wang
# perspective transformation class
# --------------------------------

from glob import glob
import cv2
import numpy as np
import os
from Helper import *

class perspective(object):
    """ perpective transformation
    """
    def __init__(self):
        self.img_size = (1280, 720)
        # source 4 locations
        self.src = np.float32([[220, 720],[580, 460],[710, 460],[1100, 720]])
        # margin size
        self.offset = 200
        # 4 destination points
        self.dst = np.float32([[self.src[0][0]+self.offset, 720],[self.src[0][0]+self.offset, 0],[self.src[-1][0]-self.offset, 0],[self.src[-1][0]-self.offset, 720]])
        # transform matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        # inverse transform matrix
        self.MV = cv2.getPerspectiveTransform(self.dst, self.src)
        self.img = np.zeros(self.img_size)
    def warped(self, image, show=False):
        """ perspective transformation for image
            @param:
                image: image to transform
                show: whether show visualization
            @output:
                transformed image
        """
        if show:
            # draw lines on image
            image = cv2.line(image, (self.src[0][0], self.src[0][1]), (self.src[1][0], self.src[1][1]),  (0,0,255), thickness=6)
            image = cv2.line(image, (self.src[-1][0], self.src[-1][1]), (self.src[-2][0], self.src[-2][1]),  (0,0,255), thickness=6)
            image = cv2.line(image, (self.src[1][0], self.src[1][1]), (self.src[-2][0], self.src[-2][1]),  (0,0,255), thickness=3)
        warped_img = cv2.warpPerspective(image, self.M, self.img_size)
        self.img = image
        if show:
            plot_dual(image, warped_img, 'threshed image','transformed image', cm1='gray',cm2='gray')
        return warped_img
    
    def unwarped(self, lanes):
        """ inverse perspective transformation
            @param: 
                lanes: image contains lanes
            @outout:
                inversed images
        """
        # Warp lane boundaries back onto original image
        lane_lines = cv2.warpPerspective(lanes, self.MV, self.img_size)
        return lane_lines
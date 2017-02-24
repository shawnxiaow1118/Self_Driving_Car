#!/usr/bin/env python
# author: Xiao Wang
# line class
# --------------------------------

import cv2
import numpy as np
from scipy.misc import imresize, imread
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Helper import *

class Line(object):
    """ Line class contains history information
    """
    def __init__(self, n_frames=1, x=None, y=None):
        """ @param:
                n_frames: number of frame to remember
                x, y: x and y pos for line
        """
        # Frame memory
        self.n_frames = n_frames
        # was the line detected in the last iteration?
        self.detected = False
        # number of pixels added per frame
        self.n_pixel_per_frame = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # Polynom for the current coefficients
        self.current_fit_poly = None
        # Polynom for the average coefficients over the last n iterations
        self.best_fit_poly = None
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
        
        
    def update(self, x_pos, y_pos):
        """ update line object using newly find positions
            @param:
                x_pos: x position of the object
                y_pos: y position of the object
        """
#         if x_pos == None:
#             if len(self.n_pixel_per_frame) > 0:
#                 num_to_remove = self.n_pixel_per_frame.pop(0)
#                 self.recent_xfitted = self.recent_xfitted[num_to_remove:]
        assert len(x_pos)==len(y_pos),'x and y have to have the same size'
        self.allx = x_pos
        self.ally = y_pos
        self.n_pixel_per_frame.append(len(self.allx)) # number of frame 
        if len(self.n_pixel_per_frame) > self.n_frames:
            num_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[num_to_remove:]
        self.recent_xfitted.extend(x_pos)
        self.bestx = np.mean(self.recent_xfitted)
        
        tmp_fit = np.polyfit(self.ally, self.allx, 2)
        self.current_fit = tmp_fit
        if self.best_fit == None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (len(self.n_pixel_per_frame)-1) + self.current_fit)/len(self.n_pixel_per_frame)
        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

        self.radius_of_curvature = cal_curvature(self.best_fit_poly)
        
    def is_parallel(self, other_line, threshold=(0, 0)):
        """ judge whether two line are near parallel 
            @param:
                other_line: another line that we wish to compare with
                threshold: thresh hold to tell whether parallel or not
            @output:
                boolean value
        """
        dif_1 = np.abs(self.current_fit[0] - other_line.current_fit[0])
        dif_2 = np.abs(self.current_fit[1] - other_line.current_fit[1])
        print("diff in angle {}, {}".format(dif_1, dif_2))

        parallel = dif_1 < threshold[0] and dif_2 < threshold[1]  # conditions

        return parallel

    def get_current_fit_distance(self, other_line):
        """ the distance between 2 lines
        """
        return np.abs(self.current_fit_poly(720) - other_line.current_fit_poly(720))

    def get_best_fit_distance(self, other_line):
        """ the distance between 2 lines
        """
        return np.abs(self.best_fit_poly(720) - other_line.best_fit_poly(720))
#!/usr/bin/env python
# author: Xiao Wang
# line detector class
# --------------------------------
from Helper import *
from Line import *
from Perspective import *
import cv2
import numpy as np
from scipy.misc import imresize, imread
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def hist_search(binary_warped, show=False):
    """ search for line points using histogram search
        @param:
            binary_warped: binary warped image 
        @output:
            leftx, lefty: points position for left line
            rightx, righty: points position for right line
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,25,255), 4) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,25,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if show:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [25, 25, 25]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [25, 25, 25]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='red', linewidth=3)
        plt.plot(right_fitx, ploty, color='yellow',linewidth=3)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    return leftx, lefty, rightx, righty


def poly_search(binary_warped, left_fit, right_fit):
    """ search for line points using predefined line fit
        @param:
            binary_warped: binary warped image 
            left_fit, right_fit: left and right line fit
        @output:
            leftx, lefty: points position for left line
            rightx, righty: points position for right line
    """

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    # narrow search region
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty


class LineDetector():
    """ line detector pipeline
    """
    def __init__(self, cliber):
        # left line object
        self.left = None
        # right line object
        self.right = None
        # is the new line fit reasonable
        self.curvature = None
        self.center_poly = None
        self.offset = None
        # perspective transformation object
        self.pers = perspective()
        # count the number of consecutive unsuccessful lines
        self.bad_lines = 0
        # calibration object
        self.cliber = cliber
    
    def check_sanity(self, leftx, lefty, rightx, righty):
        """ check the new line fitted are reasonable or not
            @param:
                leftx, lefty: left line poitns
                rightx, righty: right line points
            @output:
                boolean value
        """
        if len(leftx)==0 or len(rightx)==0:
            return False
        else:
            # left and right line object for new detected points
            new_left = Line(n_frames=1, x = leftx, y = lefty)
            new_right = Line(n_frames=1, x = rightx, y = righty)
            # update to get fitted
            new_left.update(leftx, lefty)
            new_right.update(rightx, righty)
            # check with each other
            left_right_pair = is_reasonable(new_left, new_right)
#             print("left_right {}".format(left_right_pair))
            # check with the latest line
            if self.left is not None and self.right is not None:
                left_time_pair = is_reasonable(new_left, self.left)
                right_time_pair = is_reasonable(new_right, self.right)
                plausible = left_time_pair and right_time_pair
            return plausible or left_right_pair
        
               
    def pipeline_image(self, image):
        """ process pipeline for single image
            @param:
                image: color image
            @output:
                combiend image with original and lines fitted with area colored green
        """
        # read undistored image
        img = self.cliber.undistorted(image)
        # using color and hsv selector to select the line
        thres_img = thresh(img)
        # perspective transformation
        new_img = self.pers.warped(thres_img)
        # initialization
        if self.left == None:
            leftx, lefty, rightx, righty = hist_search(new_img)
            self.left = Line(n_frames=15, x = leftx, y = lefty) # use 15 frame memory
            self.right = Line(n_frames=15, x = rightx, y = righty)
            self.left.update(leftx, lefty)
            self.right.update(rightx, righty)
        # update
        else:
            leftx, lefty, rightx, righty = poly_search(new_img, self.left.best_fit, self.right.best_fit)
            #print("length of points {}, {}".format(len(leftx), len(rightx)))
            is_sanity = self.check_sanity(leftx, lefty, rightx, righty)
            #print("sanity {}".format(is_sanity))
            if is_sanity:
                # update only if two new lines are plausible
                self.left.update(leftx, lefty)
                self.right.update(rightx, righty)
                self.bad_lines = 0
            else:
                self.bad_lines += 1
                print(self.bad_lines)
                if (self.bad_lines > 15 or len(self.left.n_pixel_per_frame) < 1):
                    leftx, lefty, rightx, righty = hist_search(new_img)
                    if len(leftx)==0 or len(rightx)==0:
                        pass
                    else:
                        # refresh line objects
                        self.left = Line(n_frames=15, x = leftx, y = lefty)
                        self.right = Line(n_frames=15, x = rightx, y = righty)
                        self.left.update(leftx, lefty)
                        self.right.update(rightx, righty)
                else:
                    pass
        empty_mask = np.zeros((720, 1280))
        color_mask = cv2.cvtColor(empty_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        area = line_area(empty_mask, self.left.best_fit, self.right.best_fit)
        color_mask[area == 1] = [0,255,0]
        lines = self.pers.unwarped(color_mask)
        combined_img = cv2.addWeighted(img, 1, lines, 0.7, 0)
        # show infromation
        if self.left is not None and self.right is not None:
            self.center_poly = (self.left.best_fit_poly + self.right.best_fit_poly) / 2
            self.curvature = cal_curvature(self.center_poly)
            self.offset = (img.shape[1] / 2 - self.center_poly(720)) * 3.7 / 700
            put_text(combined_img, self.curvature, self.offset)
#         plot_dual(img, combined_img)
#         plt.show()
        return combined_img
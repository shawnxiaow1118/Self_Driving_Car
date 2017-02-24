#!/usr/bin/env python
# author: Xiao Wang
# help functions
# --------------------------------

import cv2
import numpy as np
from scipy.misc import imresize, imread
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_dual(img1, img2, title1='', title2='', figsize=(24, 9), cm1 = None,cm2 = None):
    """ Plot two images side by side.
        @param:
            img1: first image             img2: second image
            title1: title for first image title2: title for second image 
            figsize=: tuple of 2, size of each figure
            cm1: color map for first image cm2: color map for second image
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f.tight_layout()
    ax1.imshow(img1,cmap=cm1)
    ax1.set_title(title1, fontsize=25)
    ax2.imshow(img2,cmap=cm2)
    ax2.set_title(title2, fontsize=25)
    plt.show()


def test_undistorted(image_name, cliber):
    """ test the calibration procedure
        @param:
            image_name: image file name
        @ouput:
            corresponding undistorted image
    """
    img = mpimg.imread(image_name)
    print(img.shape)
    img_un = cliber.undistorted(img)
    plot_dual(img, img_un, "original","undistorted")
    return img_un

def plot(img, title='',cm=None):
    """ plot image with title
    """
    plt.imshow(img,cmap=cm)
    plt.title(title)
    plt.show()



def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255), show=False):
    """ x or y directional gradient
        @params:
            img: image  orient: 'x' for gradient in x direction 'y' for gradient in y direction
            sobel_kernel: kernel size  thresh: threshhold for gradient filter show: visualization flag
        @output:
            binary image after threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in y
        abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from vertical
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    # Apply threshold
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    if show:
        plot_dual(img, sxbinary,'Original image','Sobel {} gradient threshold'.format(orient),cm2='gray')
    return sxbinary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255), show=False):
    """ Calculate gradient magnitude
        @params:
            img: image  sobel_kernel: kernel size  
            thresh: threshhold for gradient filter show: visualization flag
        @output:
            binary image after threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag_sobel = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    mag_binary = np.zeros_like(mag_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    if show:
        plot_dual(img, mag_binary,'Original image','Gradient magnitute threshold',cm2='gray')
    return mag_binary

def dir_threshold(img, sobel_kernel=9, thresh=(0, np.pi/2), show=False):
    """ Calculate direction of gradient
        @params:
            img: image  sobel_kernel: kernel size  
            thresh: threshhold for gradient filter show: visualization flag
        @output:
            binary image after threshold
    """
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direct = np.arctan2(np.abs(sobely), np.abs(sobelx))
    dir_binary = np.zeros_like(direct)
    dir_binary[(direct>=thresh[0])&(direct<=thresh[1])] = 1
    if show:
        plot_dual(img, dir_binary,'Original image','Gradient direction threshold',cm2='gray')
    return dir_binary


def hls_select(img, thresh=(0, 255), show=False):
    """ Calculate hls threshold for s channel 
        @params:
            img: image show: visualization flag
            thresh: threshhold for gradient filter 
        @output:
            binary image after threshold
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) # transform to hls 
    s_channel = hls[:,:,2] # extract s channel
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    if show:
        plot_dual(img, binary,'Original image','HLS threshold',cm2='gray')
    return binary

def rgb_select(img, thresh=(0, 255), show=False):
    """ Calculate hls threshold for color image
        @params:
            img: image show: visualization flag
            thresh: threshhold for gradient filter 
        @output:
            binary image after threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    if show:
        plot_dual(img, binary,'Original image','RGB threshold',cm2='gray')
    return binary

def combined_three(threshed_images, show=False):
    """ Calculate combined binary image
        @params:
            threshed_iamges: 3 binary images
            show: visualization flag
        @output:
            binary image after threshold
    """    
    combined_binary = np.zeros_like(threshed_images[0])
    combined_binary[(threshed_images[0] == 1) |((threshed_images[1] == 1)|(threshed_images[2] == 1))] = 1
    if show:
        plot(combined_binary,cm='gray')
    return combined_binary

def yellow_select(img, thresh=(0, 255), show=False):
    """ Calculate yellow threshold for color image
        @params:
            img: image show: visualization flag
            thresh: threshhold for color filter 
            show: visualization flag
        @output:
            binary image after threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # better to use hsv to extract yellow object from color image
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    binary = np.zeros_like(gray)
    binary[(hsv[:,:,0] > 10) &  (hsv[:,:,1] > 100) & (hsv[:,:,2] > 100) & (hsv[:,:,0] < 40) & (hsv[:,:,1] < 255) & (hsv[:,:,2] < 255)] = 1
    if show:
        plot_dual(img, binary,'Original image','RGB threshold',cm2='gray')
    return binary  

def thresh(img, show=False):
    """ final threshold bianry selector for line selection
        @param:
            img: image
        @output:
            combined binary image
    """
    hls = hls_select(img, thresh=(90,255))
    rgb = rgb_select(img, thresh=(210,255))
    direct = dir_threshold(img,  thresh=(0.7,1.3))
    sobel = abs_sobel_thresh(img, orient='x', thresh=(150,200))
    yellow = yellow_select(img, thresh=(20,255),show=False)
    combined_binary = np.zeros_like(direct)
    combined_binary[((hls==1)&(yellow==1)) | (rgb==1)] = 1
    if show:
        plot_dual(img, combined_binary, 'original image','threholded image',cm2='gray')
    return combined_binary


def cal_curvature(fit):
    """ calculate the curvature
        @param:
            fit: a line fit
        @output:
            curvature of the fitted line
    """
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    y_pos = np.array(np.linspace(0, 720, num=10))
    x_pos = np.array([fit(x) for x in y_pos])
    y_eval = np.max(y_pos)

    fit = np.polyfit(y_pos * ym_per_pix, x_pos * xm_per_pix, 2)
    curvature = ((1 + (2 * fit[0] * y_eval / 2. + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
    return curvature


def is_reasonable(line1, line2, para_thresh=(0.02,0.58), dist_thresh=(330,550)):
    """ two lines satisfy parallel condition or not
        @param:
            line1, line2: two line objects
            para_thresh: curvature condition
            dist_thresh: distance condition
        @output:
            boolean value
    """
    parallel = line1.is_parallel(line2, para_thresh)
    print("is paralllel {}".format(parallel))
    cur_dist = line1.get_current_fit_distance(line2)
    print("curdist {}".format(cur_dist))
    dist = (dist_thresh[0] < cur_dist) and (cur_dist < dist_thresh[1])
    return dist and parallel


# functions modified from https://github.com/pkern90/CarND-advancedLaneLines
def put_text(img, curvature, offset):
    """ put text on the image
        @param:
            curvature: curvature of the fitted line
            offset: offset to the center of the road
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
    left_or_right = 'left' if offset < 0 else 'right'
    cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(offset), left_or_right), (50, 100), font, 1,
                (255, 255, 255), 2)

def line_area(empty_mask, left_fit, right_fit, start_y=0, end_y =720):
    """ draw area between fitted lines
        @param:
            empty_mask: empty image  left_fit: left fitted line
            right_fit: right fitted line 
        @output:
            image mask with area between two lines highlighted
    """
    mask = empty_mask
    for y in range(start_y, end_y):
        left = evaluate_poly(left_fit, y)
        right = evaluate_poly(right_fit, y)
        mask[y][int(left):int(right)] = 1
    return mask


def evaluate_poly(fit, yvar):
    """ output predicted x value
        @param:
            fit: fitted poly
            yvar: y_value
        @output:
            predicted x value
    """
    output = fit[0]*yvar**2 + fit[1]*yvar + fit[2]
    return output


# Create an image to draw on and an image to show the selection window
def draw_line(binary_warped):
    """ help function to visualize
        @param:
            binary image
    """
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [25, 25, 25]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [25, 25, 25]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
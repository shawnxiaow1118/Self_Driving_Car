import numpy as np
import cv2
from scipy.ndimage.measurements import label
from helper import *


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """ get list of windows to search
        @param:
            x_start_stop, y_start_stop: list of points of the staring and ending point(x and y) of the search window on original image
            xy_window: basic window size 
            xy_overlap: overlap between consecutive search windows(horizonta and vertical)
        @ouput:
            list of windows(represent as tuples)

    """
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def add_heat(heatmap, bbox_list):
    """ get a heatmap of search windows classified as vehicle
        @param:
            heatmap: empty template
            bbox_list: found windows
        @output:
            heatmap
    """
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap
    
def apply_threshold(heatmap, threshold):
    """ zero out pixels below threshold
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    """ draw rectangle for each car founded
        @param:
            labels: tuples,first contains the array of labeled data, second is the number of features
        @output:
            image with cars boxes drawed
    """
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # manually add or subtract some pixels form the value because I used larger threshold, and there are less boxes contain the margin 
        # by addition and subtraction we can get a better view
        bbox = ((np.max((0, np.min(nonzerox)-12)), np.max((np.min(nonzeroy)-12,0))), (np.min((np.max(nonzerox)+12, img.shape[1]-1)), 
          np.min((np.max(nonzeroy)+12, img.shape[0]-1))))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show=False):
    """ Define a single function that can extract features using hog sub-sampling and make predictions
        @param:
            ystart, ystop: staring and ending postion of search windows
            scale: size of window relative to basic size(here is 64x64)
            svc: trained linear SVM classifier
            X_scaler: standard scaler used in model training
            orient: number of orientation considered  
            pix_per_cell: number of pixels in one cell
            cell_per_block: number of cells in one block
            show: if true display the result image
            spatial_size: new resolution
            hist_bins: bins to count
        @output:
            windows: all windows searched
            hot_windows: windows that are classified as cars
    """
    
    draw_img = np.copy(img)
    # original image(read in .jpg) is in scale from 0-255(np.unit8) need to rescale for that trainging was based on (0,1) scale
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # HOG features, only calculate once
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    hot_windows = []
    windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG featrues
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell+15 # add 15 to span all the vertical direction
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    

            test_prediction = svc.predict(test_features)
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
            
            if test_prediction == 1:
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                hot_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    if show:
        return draw_img
    else:
        return hot_windows, windows
                
    return draw_img


def search_scales(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show=False):
    """ search windows with different scales
        @param:
            svc: trained linear SVM classifier
            X_scaler: standard scaler used in model training
            orient: number of orientation considered  
            pix_per_cell: number of pixels in one cell
            cell_per_block: number of cells in one block
            show: if true display the result image
            spatial_size: new resolution
            hist_bins: bins to count
        @output:
            boxes contains 'cars' with different scales
    """
    # start position and correspoding scales
    ystart = [380, 380, 380]
    ystop = [500, 580, 620]
    scale = [1.2, 1.4, 1.7]
    draw_image = np.copy(img)
    all_windows = []
    hot_windows = []
    for i in range(len(scale)):
        hot_win, win = find_cars(img, ystart[i], ystop[i], scale[i], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        all_windows.extend(win)
        hot_windows.extend(hot_win)
        if show:
            window_img = draw_boxes(draw_image, win, color=(20*i, 5*i, i*30), thick=4)  
            plt.imshow(window_img)
            plt.show()
    print("total window searched is {}".format(len(all_windows)))
    print("total hot window found is {}".format(len(hot_windows)))
    if show:
        plt.show()
    return hot_windows
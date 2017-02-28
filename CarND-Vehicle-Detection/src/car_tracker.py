import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import pickle

from helper import *
from windows import *

class Tracker(object):
    """ tracker obkect to track cars on video stream
    """
    def __init__(self, pickle_file, n_frames = 8):
        """ initialization
            @param:
                n_frames: frame history to store
                pickle_file: models to load
        """
        self.n_frames = n_frames
        # all boxes classified as cars
        self.boxes = []
        # the length of windoes within each frame found
        self.lengths = []
        # load trained model information
        self.dist_pickle = pickle.load( open(pickle_file, "rb" ) )
        # svc: classifier, X_scaler: standard scaler
        self.svc = self.dist_pickle["svc"]
        self.X_scaler = self.dist_pickle["X_scaler"]
        self.orient = self.dist_pickle["orient"]
        self.pix_per_cell = self.dist_pickle["pix_per_cell"]
        self.cell_per_block = self.dist_pickle["cell_per_block"]
        self.spatial_size = self.dist_pickle["spatial_size"]
        self.hist_bins = self.dist_pickle["hist_bins"]
        self.colorspace = self.dist_pickle["color_space"]
        
        
        
    def update(self, new_boxes):
        """ update self boexes with new frame's boxes
            @param:
                new_boxes: new boxes classified as cars on new frame
        """
        if len(self.lengths) > self.n_frames:
            # delete oldest if exceed capacity
            del self.boxes[0:self.lengths[0]]
            del self.lengths[0]
        self.lengths.append(len(new_boxes))
        self.boxes.extend(new_boxes)
        
    def detect(self, img, thresh_hold, show=False):
        """ detect cars on image using historical information and heatmap
            @param:
                thresh_hold: thresh_hold for one frame
            @output: 
                image with detected cars
        """
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # heatmap for all boxes found in last n_frames frame
        heat = add_heat(heat,self.boxes)
        # threshold for multiple frames
        thresh = thresh_hold*len(self.lengths)
        # initialization
        if len(self.lengths) == 0:
            thresh = thresh_hold
        heat = apply_threshold(heat,thresh)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        if show:
            plot_dual(draw_img, heatmap,'Car positions','Heat Map',cm2='hot')
        return draw_img
    
    def process_image(self, img):
        """ pipeline for new frame image
        """
        show = False
        # draw_image = np.copy(img)
        # search all scale windows and return cars' windows
        hots = search_scales(img,self.svc, self.X_scaler, self.orient, 
                             self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)
        # update the self boxes
        self.update(hots)
        # detect cars using threshold
        window_image = self.detect(img, 2)
        if show:
            plt.imshow(window_image)
            plt.show()
        return window_image
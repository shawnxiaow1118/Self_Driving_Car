import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt


def convert_color(img, conv='RGB2YCrCb'):
  """ convert image to specific color space
      @param:
        img: image    conv: color transforamtion 
      @output:
        transformed color image
  """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, show=False, feature_vec=True):
  """ Get HOG features from image
      @param:
        img: image orient: number of orientation considered  
        pix_per_cell: number of pixels in one cell
        cell_per_block: number of cells in one block
        show: if true return hog features image
        feature_vec: if true return features
      @output:
        HOG feature vector
  """
    if show == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
  """ spatial features, low resolution
      @param:
        img: image size: new resolution
      @output:
        spatial feature vector
  """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
  """ color histogram feature
      @param:
        img: image nbins: bins to count
      @output:
        histogram feature vector
  """
    # Compute the histogram of the each color channel
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def plot(img, title=''):
  """ plot image
  """
    plt.imshow(img)
    plt.title(title)
    plt.show()


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  """ draw rectangles on img according to bboxes
      @param:
        bboxes: list of boxes ((x1,y1),(x2,y2))
        color, thick: parameters for the plot setting
      @output:
        image with rectangles
  """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


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



def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """ extract features from an imgs, used for training and testing
        @param:
            color_space: color space to use 
            spatial_size: new resulotion
            hist_bin: bins to count
            orient: number of orientation of HOG features
            pix_per_cell,cell_per_block: HOG features parameters
            hog_channel: hog features of which channel
    """
    features = []
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
    return features
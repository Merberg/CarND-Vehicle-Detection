import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Functions for vehicle detect features extraction

def convertColorSpace(img, color_space_in):
    if color_space_in == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if color_space_in == 'BGR':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
def extractHog(img, orient=9, pix_per_cell=8, n_cell_per_block=2, 
               vis=False, feature_vec=True):
    # Call with two outputs if an image of the HOG is requested
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(n_cell_per_block, n_cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(n_cell_per_block, n_cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def binColorsSpatially(img, size=(8, 8)):
    # Down sample the image and stack
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def computeColorHistogram(img, n_bins=40):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=n_bins)
    channel2_hist = np.histogram(img[:,:,1], bins=n_bins)
    channel3_hist = np.histogram(img[:,:,2], bins=n_bins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features               
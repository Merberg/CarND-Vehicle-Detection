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
    
def extractHog(img, orient, pix_per_cell, n_cell_per_block, 
               vis=False, feature_vec=True):
    # Call with two outputs if an image of the HOG is requested
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(n_cell_per_block, n_cell_per_block), 
                                  block_norm='L2-Hys', transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(n_cell_per_block, n_cell_per_block), 
                       block_norm='L2-Hys', transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def binColorsSpatially(img, size):
    # Down sample the image and stack
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def computeColorHistogram(img, n_bins):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=n_bins)
    channel2_hist = np.histogram(img[:,:,1], bins=n_bins)
    channel3_hist = np.histogram(img[:,:,2], bins=n_bins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extractImgFeatures(img, color_space_in, hog_orient, hog_pix_per_cell, 
                       hog_cells_per_block, spatial_size, hist_bins, 
                       vis, hog_channel):
    # Extract the features from the image
    img_features = []
    imgC = convertColorSpace(img, color_space_in)

    # First the HOG
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(imgC.shape[2]):
            hog = extractHog(imgC[:,:,channel], hog_orient, hog_pix_per_cell, 
                             hog_cells_per_block, vis=False, feature_vec=True)
            hog_features.append(hog)
        hog_features = np.ravel(hog_features)
    else:
        if vis == True:
            hog_features, hog_image = extractHog(imgC[:,:,hog_channel], hog_orient, 
                                                 hog_pix_per_cell, hog_cells_per_block,
                                                 vis=True, feature_vec=True)
        else:
            hog_features = extractHog(imgC[:,:,hog_channel], hog_orient, 
                                                 hog_pix_per_cell, hog_cells_per_block,
                                                 vis=False, feature_vec=True)
    img_features.append(hog_features)

    # Second the color binning
    spatial_features = binColorsSpatially(imgC, spatial_size)
    img_features.append(spatial_features)

    # Third the color histogram
    hist_features = computeColorHistogram(imgC, hist_bins)
    img_features.append(hist_features)

    # Concatenate features and return
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)
    
def extractFeatures(imgs, color_space_in, hog_orient, hog_pix_per_cell, 
                    hog_cells_per_block, spatial_size, hist_bins):
    # Extract features from a list of images
    features = []
    for img in imgs:
        img_features = extractImgFeatures(img, color_space_in, 
                                          hog_orient, hog_pix_per_cell, hog_cells_per_block,
                                          spatial_size, hist_bins, 
                                          vis=False, hog_channel='ALL')
        
        # Concatenate features and append to the imgs' list
        features.append(img_features)
    return features
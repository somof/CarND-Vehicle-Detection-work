import numpy as np
import cv2
from skimage.feature import hog


def convert_color(img, conv='RGB'):
    if conv == 'RGB':
        return np.copy(img)
    elif conv == 'HSV' or conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == 'LUV' or conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'HLS' or conv == 'RGB2HSLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == 'YUV' or conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif conv == 'YCrCb' or conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == 'BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif conv == 'BGR2LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif conv == 'BGR2HSLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif conv == 'BGR2YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     transform_sqrt=True, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis is True:
        print('never reached code!')
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=transform_sqrt,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=transform_sqrt,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    # print('  spat: ', np.max(img))
    # Use cv2.resize().ravel() to create the feature vector
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=64, hist_range=(0, 1.0)):
    # print('  hist: ', np.max(img))
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=hist_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=hist_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=hist_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def single_img_features(image,
                        color_space='YCrCb',
                        spatial_size=(32, 32),
                        hist_bins=64, hist_range=(0, 255),
                        transform_sqrt=False,
                        orient=9, pix_per_cell=8, cell_per_block=2):
    features = []
    # 1) Apply color conversion
    img = convert_color(image, conv=color_space)

    # 2) Compute spatial features
    spatial_features = bin_spatial(img, size=spatial_size)
    features.append(spatial_features)

    # 3) Compute histogram features
    hist_features = color_hist(img, nbins=hist_bins, hist_range=hist_range)
    features.append(hist_features)

    # 4) Compute HOG features
    hog_features = []
    for channel in range(img.shape[2]):
        hog_features.extend(get_hog_features(img[:, :, channel],
                                             orient, pix_per_cell, cell_per_block,
                                             transform_sqrt=transform_sqrt,
                                             vis=False, feature_vec=True))
    # print(len(hog_features))
    features.append(hog_features)

    # print('spa ', spatial_features.shape)  # 768 = (16, 16) x 3ch
    # print('his ', hist_features.shape)  # 96 = 32bins x 3ch
    # print('hog ', len(hog_features))  # 5292 = 7x7 x 4(?) x 9 x 3ch
    # exit(0)
    #  Feature vector length: 16740
    #    using: 9 orientations 8 pixels per cell and 2 cells per block

    # training
    # Feature vector length: 16740
    # 16740 - 768 - 96 = 15876 = 5292 x 3

    return np.concatenate(features)

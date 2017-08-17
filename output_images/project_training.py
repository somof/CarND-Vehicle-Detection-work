import matplotlib.image as mpimg
import numpy as np
import os
import cv2
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions_training import single_img_features


use_small_number_sample = False  # True
use_smallset = False

filename = 'svc_pickle.'
if use_smallset:
    filename += 'smallset.'
if use_small_number_sample:
    filename += 'small.'
filename = filename + 'p'

if os.path.exists(filename):
    print('pickle file already exists. : ', filename)
    exit(0)

# Tweak these parameters and see how the results change.
color_space    = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9  # HOG orientations
pix_per_cell   = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
spatial_size   = (32, 32)  # Spatial binning dimensions
hist_bins      = 64  # Number of histogram bins


# Read in cars and notcars
cars = []
notcars = []
if use_smallset:
    for file in glob.glob('../src/dataset/*_smallset/*/*.jpeg'):
        if 'image' in file or 'extra' in file:
            notcars.append(file)
        else:
            cars.append(file)
else:
    for file in glob.glob('../src/dataset/vehicles/*/*.png'):
        cars.append(file)
    for file in glob.glob('../src/dataset/non-vehicles/*/*.png'):
        notcars.append(file)

print('cars data length:', len(cars))
print('notcars data length:', len(notcars))

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
if use_small_number_sample:
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]


def extract_features(imgs,
                     color_space='YCrCb',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        img = mpimg.imread(file)
        if not use_smallset:
            img = img * 255
            img = img.astype(np.uint8)
        hist_range = (0, 255)

        img_features = single_img_features(img,
                                           color_space=color_space,
                                           spatial_size=spatial_size,
                                           hist_bins=hist_bins, hist_range=hist_range,
                                           orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
        features.append(img_features)

    # Return list of feature vectors
    return features


car_features = extract_features(cars,
                                color_space=color_space,
                                spatial_size=spatial_size,
                                hist_bins=hist_bins,
                                orient=orient,
                                pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block)

notcar_features = extract_features(notcars,
                                   color_space=color_space,
                                   spatial_size=spatial_size,
                                   hist_bins=hist_bins,
                                   orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Training/Test data size:', len(y_train), len(y_test))
print('Feature vector length:', len(X_train[0]))
print('  using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')


# Use a linear SVC
print('Training via SVC')
svc = LinearSVC()

t = time.time()  # Check the training time for the SVC
svc.fit(X_train, y_train)
t2 = time.time()

print('  ', round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('  Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


dist_pickle = {}
dist_pickle["color_space"] = color_space
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins

pickle.dump(dist_pickle, open(filename, "wb"))

print('Stored into pickle:')
print('  color_space: ', color_space)
print('  svc: ', svc)
print('  X_scaler: ', X_scaler)
print('  orient: ', orient)
print('  pix_per_cell: ', pix_per_cell)
print('  cell_per_block: ', cell_per_block)
print('  spatial_size: ', spatial_size)
print('  hist_bins: ', hist_bins)

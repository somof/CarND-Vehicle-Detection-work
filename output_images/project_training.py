
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions_training import single_img_features


use_float_image = False  # True
use_small_sample = False  # True


# Tweak these parameters and see how the results change.
color_space    = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9  # HOG orientations
pix_per_cell   = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
spatial_size   = (32, 32)  # (16, 16)  # Spatial binning dimensions
hist_bins      = 64  # 32  # Number of histogram bins


# Read in cars and notcars
cars = []
notcars = []
for file in glob.glob('../src/dataset/*_smallset/*/*.jpeg'):
    if 'image' in file or 'extra' in file:
        notcars.append(file)
    else:
        cars.append(file)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
if use_small_sample:
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
        hist_range = (0, 255)
        if use_float_image:
            img = img.astype(np.float32) / 255
            hist_range = (0, 1.0)

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
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
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

filename = 'svc_pickle.'
if use_float_image:
    filename += 'float.'
if use_small_sample:
    filename += 'small.'

pickle.dump(dist_pickle, open(filename + 'p', "wb"))

print('Stored into pickle:')
print('  color_space: ', color_space)
print('  svc: ', svc)
print('  X_scaler: ', X_scaler)
print('  orient: ', orient)
print('  pix_per_cell: ', pix_per_cell)
print('  cell_per_block: ', cell_per_block)
print('  spatial_size: ', spatial_size)
print('  hist_bins: ', hist_bins)


# Search Parameter
y_start_stop = [400, 720]  # Min and max in y to search in slide_window()

print('Search Parameter:')
print('  y_start_stop:', y_start_stop)


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


def search_windows(image, windows, clf, scaler,
                   color_space='YCrCb',
                   spatial_size=(32, 32),
                   hist_bins=32, hist_range=(0, 255),
                   orient=9, pix_per_cell=8, cell_per_block=2):

    on_windows = []
    for window in windows:

        # 1) Extract the test window from original image
        img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 2) Extract features for that window using single_img_features()
        img_features = single_img_features(img,
                                           color_space=color_space,
                                           spatial_size=spatial_size,
                                           hist_bins=hist_bins, hist_range=hist_range,
                                           orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

        # 3) Scale extracted features to be fed to classifier
        features = scaler.transform(np.array(img_features).reshape(1, -1))

        # 4) Predict using your classifier
        prediction = clf.predict(features)
        if prediction == 1:
            on_windows.append(window)

    return on_windows


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Draw a rectangle given bbox coordinates
    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img


for no in range(1, 7):
    file = '../test_images/test{}.jpg'.format(no)
    print(file)
    img = mpimg.imread(file)
    hist_range = (0, 255)
    bbox_color = (0, 0, 255)
    if use_float_image:
        img = img.astype(np.float32) / 255
        hist_range = (0, 1.0)
        bbox_color = (0, 0, 1)

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(img,
                                 windows, svc, X_scaler,
                                 color_space=color_space,
                                 spatial_size=spatial_size,
                                 hist_bins=hist_bins, hist_range=hist_range,
                                 orient=orient,
                                 pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block)

    draw_boxes(img, hot_windows, color=bbox_color, thick=4)
    plt.imshow(img)
    plt.pause(1.001)

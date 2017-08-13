import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from functions_training import single_img_features

use_float_image = False  # True
use_small_sample = True  # False

# Search Parameter
y_start_stop = [400, 720]  # Min and max in y to search in slide_window()

filename = 'svc_pickle.'
if use_float_image:
    filename += 'float.'
if use_small_sample:
    filename += 'small.'

# dist_pickle    = pickle.load(open("svc_pickle.p", "rb"))
dist_pickle    = pickle.load(open(filename + 'p', "rb"))
color_space    = dist_pickle["color_space"]
svc            = dist_pickle["svc"]
X_scaler       = dist_pickle["scaler"]
orient         = dist_pickle["orient"]
pix_per_cell   = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size   = dist_pickle["spatial_size"]
hist_bins      = dist_pickle["hist_bins"]

print('Restored into pickle:')
print('  color_space: ', color_space)
print('  svc: ', svc)
print('  X_scaler: ', X_scaler)
print('  orient: ', orient)
print('  pix_per_cell: ', pix_per_cell)
print('  cell_per_block: ', cell_per_block)
print('  spatial_size: ', spatial_size)
print('  hist_bins: ', hist_bins)

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


def search_windows(img, windows, clf, scaler,
                   color_space='YCrCb',
                   spatial_size=(32, 32),
                   hist_bins=32, hist_range=(0, 255),
                   orient=9, pix_per_cell=8, cell_per_block=2):

    on_windows = []
    for window in windows:

        # 1) Extract the test window from original image
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 2) Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins, hist_range=hist_range,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

        # 3) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 4) Predict using your classifier
        prediction = clf.predict(test_features)
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
    plt.pause(.001)

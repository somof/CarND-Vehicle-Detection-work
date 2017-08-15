import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle
from scipy.ndimage.measurements import label
from functions_vehicle import find_cars_multiscale
from functions_vehicle import set_perspective_matrix

use_float_image = False  # True
use_small_sample = False  # True

filename = 'svc_pickle.'
if use_float_image:
    filename += 'float.'
if use_small_sample:
    filename += 'small.'
filename = filename + 'p'

dist_pickle    = pickle.load(open(filename, "rb"))
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


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def process_image(image, weight=0.5):

    # 8) Vehicles Detection
    draw_img = np.copy(image)
    draw_img, bbox_list = find_cars_multiscale(image, draw_img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    for bbox in bbox_list:
        # print('bbox: ', bbox)
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 1)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    add_heat(heatmap, bbox_list)
    heatmap = apply_threshold(heatmap, 1)
    img_heatmap = np.clip(heatmap, 0, 255)
    labels = label(img_heatmap)
    draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)

    # return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return draw_img


######################################
# set Perspective Matrix

set_perspective_matrix()


######################################
# process frame by frame for developing

clip1 = VideoFileClip('../test_video.mp4')
frameno = 0
for frame in clip1.iter_frames():
    if frameno % 2 == 0:
        print('frameno: {:5.0f}'.format(frameno))
        result = process_image(frame)
        cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    frameno += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


clip1 = VideoFileClip('../project_video.mp4')
frameno = 0
for frame in clip1.iter_frames():
    if 160 < frameno and frameno % 4 == 0:
        print('frameno: {:5.0f}'.format(frameno))
        result = process_image(frame)
        cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    frameno += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


# for file in ('test_video.mp4', ):
# for file in ('test_video.mp4', 'project_video.mp4'):
#     clip1 = VideoFileClip('../' + file)

#     frameno = 0
#     for frame in clip1.iter_frames():
#         if frameno % 4 == 0 and frameno < 1500:
#             # print('frameno: {:5.0f}'.format(frameno))
#             result = process_image(frame)
#             cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
#             # if frameno % 10 == 0:
#             #     name, ext = os.path.splitext(os.path.basename(file))
#             #     filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
#             #     if not os.path.exists(filename):
#             #         cv2.imwrite(filename, img)
#         frameno += 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cv2.destroyAllWindows()

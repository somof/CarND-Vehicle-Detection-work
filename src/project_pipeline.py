
import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle

from functions_vehicle import *
# from functions_heatmap import *


color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 12  # 16    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

ystart = 400
ystop = 656
scale = 1.5

print('svc: ', svc)
print('X_scaler: ', X_scaler)
print('orient: ', orient)
print('pix_per_cell: ', pix_per_cell)
print('cell_per_block: ', cell_per_block)
print('spatial_size: ', spatial_size)
print('hist_bins: ', hist_bins)


def process_image(image, weight=0.5):

    # 8) Vehicles Detection
    result = find_cars(image, ystart, ystop, scale, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


######################################
# process frame by frame for developing

# for file in ('project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4'):
# for file in ('test_video.mp4', ):
for file in ('test_video.mp4', 'project_video.mp4'):
    clip1 = VideoFileClip('../' + file)

    frameno = 0
    for frame in clip1.iter_frames():
        if frameno % 4 == 0 and frameno < 1500:
        # if (frameno % 4 == 0 and
        #     ((470 <= frameno and frameno < 610) or
        #      (950 <= frameno and frameno < 1150))):

            # print('frameno: {:5.0f}'.format(frameno))

            result = process_image(frame)
            img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', img)

            # if frameno % 10 == 0:
            #     name, ext = os.path.splitext(os.path.basename(file))
            #     filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
            #     if not os.path.exists(filename):
            #         cv2.imwrite(filename, img)
            # if frameno == 300:
            #     name, ext = os.path.splitext(os.path.basename(file))
            #     filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
            #     cv2.imwrite(filename, img)
            #     exit(0)
        frameno += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

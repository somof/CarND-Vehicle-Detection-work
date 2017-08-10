import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle

from functions_vehicle import *
# from functions_heatmap import *

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
    # result = find_cars(image, ystart, ystop, scale, svc, X_scaler,
    #                    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    result = find_cars_heatmap(image, ystart, ystop, scale, svc, X_scaler,
                               orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


######################################
# process frame by frame for developing

# for file in ('test_video.mp4', ):
for file in ('test_video.mp4', 'project_video.mp4'):
    clip1 = VideoFileClip('../' + file)

    frameno = 0
    for frame in clip1.iter_frames():
        if frameno % 4 == 0 and frameno < 1500:
            # print('frameno: {:5.0f}'.format(frameno))
            result = process_image(frame)
            cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            # if frameno % 10 == 0:
            #     name, ext = os.path.splitext(os.path.basename(file))
            #     filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
            #     if not os.path.exists(filename):
            #         cv2.imwrite(filename, img)
        frameno += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

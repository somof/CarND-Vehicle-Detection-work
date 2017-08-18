import os
import cv2
import pickle
import time
import numpy as np
from moviepy.editor import VideoFileClip
from functions_vehicle import set_search_area
from functions_vehicle import find_cars_multiscale
from functions_vehicle import select_bbox_with_heatmap
from functions_vehicle import reset_hetmap_fifo
from functions_vehicle import overlay_heatmap_fifo
from functions_vehicle import overlay_heatmap_fifo_gaudy
from functions_vehicle import overlay_search_area
from functions_vehicle import reset_car_positions
from functions_vehicle import hold_car_positions
from functions_vehicle import draw_car_positions

use_small_number_sample = False
use_smallset = False

filename = 'svc_pickle.'
if use_smallset:
    filename += 'smallset.'
if use_small_number_sample:
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
transform_sqrt = dist_pickle["transform_sqrt"]

print('Restored into pickle:')
print('  color_space: ', color_space)
print('  svc: ', svc)
print('  X_scaler: ', X_scaler)
print('  orient: ', orient)
print('  pix_per_cell: ', pix_per_cell)
print('  cell_per_block: ', cell_per_block)
print('  spatial_size: ', spatial_size)
print('  hist_bins: ', hist_bins)
print('  transform_sqrt: ', transform_sqrt)


def process_image(image):

    # 8) Vehicles Detection
    draw_img = np.copy(image)
    t1 = time.time()  # Check the training time for the SVC

    # 8-1) Sliding Windows Search
    bbox_list = []
    bbox_list = find_cars_multiscale(image, svc, X_scaler, transform_sqrt, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # 8-2) Update Heatmap
    labelnum, contours, centroids = select_bbox_with_heatmap(image, bbox_list, threshold=18)  # 16 - 20

    # 8-3) Update Car Positions
    hold_car_positions(contours, centroids)

    t2 = time.time()
    print('  ', round(t2 - t1, 2), 'Seconds to process a image')

    # X) Drawing
    # display search area
    draw_img = overlay_search_area(draw_img)

    # display each detected bboxes
    # tmp_heatmap = np.zeros(draw_img.shape)
    for bbox in bbox_list:
        # print('bbox: ', bbox)
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 255), 1)
        # tmp_heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], 0] += 40
    # tmp_heatmap = np.clip(tmp_heatmap, 0, 255).astype(np.uint8)
    # draw_img = cv2.addWeighted(draw_img, 0.5, tmp_heatmap, 0.5, 0)

    # X) Overlay Vehicle BBoxes
    # for nlabel in range(1, labelnum):
    #     x, y, w, h, size = contours[nlabel]
    #     xg, yg = centroids[nlabel]
    #     cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    #     cv2.circle(draw_img, (int(xg), int(yg)), 30, (0, 0, 255), -1)
    #     cv2.putText(draw_img, '{}'.format(nlabel), (int(xg - 10), int(yg + 10)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

    # X) Draw mini Heatmap
    draw_img = overlay_heatmap_fifo_gaudy(draw_img, image, px=10, py=90, size=(180, 100))
    # draw_img = overlay_heatmap_fifo(draw_img, px=10, py=90, size=(180, 100))
    # draw_img = draw_car_positions(draw_img, px=1000, py=90)

    return draw_img


######################################
# set Perspective Matrix

set_search_area()


######################################
# process frame by frame for developing

# clip1 = VideoFileClip('../test_video.mp4')
# frameno = 0
# reset_hetmap_fifo()
# reset_car_positions()
# for frame in clip1.iter_frames():
#     if frameno % 2 == 0:
#         print('frameno: {:5.0f}'.format(frameno))
#         result = process_image(frame)
#         img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#         cv2.imshow('frame', img)
#         if frameno % 10 == 0:
#             filename = '{}_{:04.0f}fr.jpg'.format('test_video', frameno)
#             # if not os.path.exists(filename):
#             # cv2.imwrite(filename, img)
#     frameno += 1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()


clip1 = VideoFileClip('../project_video.mp4')
frameno = 0
reset_hetmap_fifo()
reset_car_positions()
for frame in clip1.iter_frames():
    if 160 < frameno and frameno % 10 == 0:
        print('frameno: {:5.0f}'.format(frameno))
        result = process_image(frame)
        img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.putText(img, 'frame {}'.format(frameno), (50, 710), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        cv2.imshow('frame', img)
        if frameno == 350:
            filename = '{}_{:04.0f}fr.jpg'.format('project_video', frameno)
            # if not os.path.exists(filename):
            cv2.imwrite(filename, img)
            # cv2.imwrite(filename, img[300:720, :, :])
            cv2.waitKey(1000)
            exit(0)
    frameno += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

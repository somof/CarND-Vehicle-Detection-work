import os
import numpy as np
import cv2
import pickle
import time
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
# from functions_vehicle import find_cars_multiscale
# from functions_vehicle import set_perspective_matrix
from functions_training import convert_color
from functions_training import get_hog_features
from functions_training import bin_spatial
from functions_training import color_hist


use_small_number_sample = False  # True

filename = 'svc_pickle.'
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

print('Restored into pickle:')
print('  color_space: ', color_space)
print('  svc: ', svc)
print('  X_scaler: ', X_scaler)
print('  orient: ', orient)
print('  pix_per_cell: ', pix_per_cell)
print('  cell_per_block: ', cell_per_block)
print('  spatial_size: ', spatial_size)
print('  hist_bins: ', hist_bins)


# M = []
# Minv = []
# search_area = []
VEHICLE_HEIGHT = 1.65


def set_perspective_matrix():

    global M, Minv, search_area

    # Calculate the Perspective Transformation Matrix and its invert Matrix
    perspective_2d = np.float32([[585, 460], [695, 460], [1127, 685], [203, 685]])
    perspective_3d = np.float32([[-1.85, 30], [1.85, 30], [1.85, 3], [-1.85, 3]])
    perspective_2d = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial
    perspective_3d = np.float32([[-1.85, 50], [1.85, 50], [1.85, 5], [-1.85, 5]])
    M = cv2.getPerspectiveTransform(perspective_3d, perspective_2d)
    Minv = cv2.getPerspectiveTransform(perspective_2d, perspective_3d)

    print('Search Area:')
    search_area = []
    for y in range(6, 24, 1):
        x = -10
        x0 = (M[0][0] * x + M[0][1] * y + M[0][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])
        y0 = (M[1][0] * x + M[1][1] * y + M[1][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])
        #
        x = 10
        x1 = (M[0][0] * x + M[0][1] * y + M[0][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])
        y1 = (M[1][0] * x + M[1][1] * y + M[1][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])
        #
        search_area.append([[int(x0), int(y0)], [int(x1), int(y1)]])
        print('{:3.0f} : ({:+8.1f},{:+8.1f}) - ({:+8.1f},{:+8.1f})'.format(y, x0, y0, x1, y1))


def find_cars_multiscale(image, draw_img, svc, X_scaler,
                         orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    global search_area

    bbox_list = []
    for area in search_area:
        scale = 1.5

        width = area[1][0] - area[0][0]
        height = int(VEHICLE_HEIGHT * width / 20)

        xstart = max(0, area[0][0])
        xstop = min(1279, area[1][0])
        ystop = area[0][1]
        ystart = ystop - height

        # print('baseline: ({:4.0f}, {:4.0f}) - ({:4.0f}, {:4.0f})  <- '.format(xstart, ystart, xstop, ystop), area)
        cv2.rectangle(draw_img, (xstart, ystart), (xstop, ystop), (255, 0, 0), 1)

        draw_img, bbox = find_cars(image, draw_img, ystart, ystop, xstart, xstop, scale, svc, X_scaler,
                                   orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if bbox:
            bbox_list.extend(bbox)

    return draw_img, bbox_list


def find_cars(img, draw_img,
              ystart, ystop, xstart, xstop, scale,
              svc, X_scaler,
              orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):

    # draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255
    # heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

    img_tosearch = img[ystart:ystop, xstart:xstop, :]

    # if 128 < img_tosearch.shape[0]:
    #     pre_scale = 128 / img_tosearch.shape[0]
    #     img_tosearch = cv2.resize(img_tosearch,
    #                               (np.int(img_tosearch.shape[1] * pre_scale),
    #                                np.int(img_tosearch.shape[0] * pre_scale)))
    #     # print(img_tosearch.shape)

    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale),
                                      np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1  # 160 - 1 = 159
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1  # 90  - 1 = 89 | 32 - 1 = 31
    # nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1  # 7
    cells_per_step = 2  # 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step  # 76
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step  # 82 | 12

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, transform_sqrt=True, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, transform_sqrt=True, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, transform_sqrt=True, feature_vec=False)

    bbox = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # Add heatmap to each box
                bbox.append([[xbox_left, ytop_draw + ystart],
                             [xbox_left + win_draw, ytop_draw + win_draw + ystart]])

    return draw_img, bbox


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

    t1 = time.time()  # Check the training time for the SVC

    # 8) Vehicles Detection
    draw_img = np.copy(image)
    draw_img, bbox_list = find_cars_multiscale(image, draw_img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    for bbox in bbox_list:
        # print('bbox: ', bbox)
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 1)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float32)
    add_heat(heatmap, bbox_list)
    heatmap = apply_threshold(heatmap, 1)

    t2 = time.time()
    print('  ', round(t2 - t1, 2), 'Seconds to process a image')

    # X) Overlay heat-map
    img_heatmap = np.clip(heatmap, 0, 255)
    labels = label(img_heatmap)
    draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)

    # X) Draw heat-map
    font_size = 1
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(draw_img, 'Heatmap', (20, 40), font, font_size, (255, 255, 255))

    mini_heatmap = np.clip(heatmap * 10 + 20, 20, 240)
    mini_heatmap = cv2.resize(mini_heatmap, (320, 180))
    mini_heatmap = cv2.cvtColor(mini_heatmap, cv2.COLOR_GRAY2RGB)
    draw_img[50:50 + mini_heatmap.shape[0], 10:10 + mini_heatmap.shape[1]] = mini_heatmap

    # X) Draw Detected car positions



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
    if frameno % 1 == 0:
        print('frameno: {:5.0f}'.format(frameno))
        result = process_image(frame)
        cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    frameno += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
exit(0)

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

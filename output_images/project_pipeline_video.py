
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle
from scipy.ndimage.measurements import label
from functions_training import convert_color
from functions_training import get_hog_features
from functions_training import bin_spatial
from functions_training import color_hist
# from functions_vehicle import *
from functions_lane import *


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
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


############################################################
# Detect Vehicle functions

VEHICLE_HEIGHT = 1.4  # meter
DISTANCE       = 30  # meter
DISTANCE_STEP  = 1  # meter
DISTANCE_NUM   = 8  # DISTANCE // DISTANCE_STEP
LANE_NUM       = 5
CENTER_LANE    = 2  # Center Lane No
FRAMENUM       = 5  # FRAMENO_0 is the current frame

car_positions = np.zeros((FRAMENUM, DISTANCE_NUM, LANE_NUM), dtype=np.uint8)
heatmap_fifo = np.zeros((FRAMENUM, 720, 1280), dtype=np.uint8)
distance_map = (7, 8, 9, 10, 11, 13, 17, 23, 30)


def set_perspective_matrix():

    global M2, M2inv, search_area

    # Calculate the Perspective Transformation Matrix and its invert Matrix
    perspective_2d = np.float32([[585, 460], [695, 460], [1127, 685], [203, 685]])
    perspective_3d = np.float32([[-1.85, 30], [1.85, 30], [1.85, 3], [-1.85, 3]])

    perspective_2d = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial
    perspective_3d = np.float32([[-1.85, 50], [1.85, 50], [1.85, 5], [-1.85, 5]])

    perspective_2d = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial
    perspective_3d = np.float32([[-1.85, 40], [1.85, 40], [1.85, 5], [-1.85, 5]])

    M2 = cv2.getPerspectiveTransform(perspective_3d, perspective_2d)
    M2inv = cv2.getPerspectiveTransform(perspective_2d, perspective_3d)

    print('Search Area:')
    search_area = []
    # for y in range(6, DISTANCE, DISTANCE_STEP):
    for y in distance_map:
        x = -1.85 - 3.7 - 3.7
        x = -10
        x0 = (M2[0][0] * x + M2[0][1] * y + M2[0][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        y0 = (M2[1][0] * x + M2[1][1] * y + M2[1][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        #
        x = 1.85 + 3.7 + 3.7
        x = 10
        x1 = (M2[0][0] * x + M2[0][1] * y + M2[0][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        y1 = (M2[1][0] * x + M2[1][1] * y + M2[1][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        #
        search_area.append([[int(x0), int(y0)], [int(x1), int(y1)]])
        print('{:3.0f} : ({:+8.1f},{:+8.1f}) - ({:+8.1f},{:+8.1f})'.format(y, x0, y0, x1, y1))


# def hold_car_positions(bbox_list):
#     global M2inv, car_positions
#     # car_positions[FRAMENUM][LANE_NUM][DISTANCE]

#     car_positions[1:FRAMENUM, :, :] = car_positions[0:FRAMENUM - 1, :, :]
#     car_positions[0][:][:] = np.zeros((1, DISTANCE_NUM, LANE_NUM), dtype=np.uint8)

#     for box in bbox_list:
#         x = (box[0][0] + box[1][0]) / 2
#         y = max(box[0][1], box[1][1])
#         x3d = (M2inv[0][0] * x + M2inv[0][1] * y + M2inv[0][2]) / (M2inv[2][0] * x + M2inv[2][1] * y + M2inv[2][2])
#         y3d = (M2inv[1][0] * x + M2inv[1][1] * y + M2inv[1][2]) / (M2inv[2][0] * x + M2inv[2][1] * y + M2inv[2][2])


#         # 5lane: -9.25 ... 9.25
#         laneno = int((x3d + 9.25) / 3.7 + 0.5)
#         distance = int(y3d / DISTANCE_STEP + 0.5)
#         # if LANE_NUM <= laneno:
#         #     print('lane no is ', laneno, ' >= ', LANE_NUM)
#         # if DISTANCE_NUM <= distance:
#         #     print('distance is ', distance * DISTANCE_STEP, ' >= ', DISTANCE)

#         # print(box, ' -> ', x, y, ' -> ', x3d, y3d, ' -> ', laneno, distance)
#         if 0 <= distance and distance < DISTANCE_NUM and 0 <= laneno and laneno < LANE_NUM:
#             car_positions[0][distance][laneno] += 1
#             if distance < DISTANCE_NUM - 1:
#                 car_positions[0][distance + 1][laneno] += 1
#             if distance < DISTANCE_NUM - 2:
#                 car_positions[0][distance + 2][laneno] += 1


def find_cars_multiscale(image, draw_img, svc, X_scaler,
                         orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    global search_area

    hog1 = None
    hog2 = None
    hog3 = None

    bbox_list = []
    for area in search_area:
        scale = 1.5

        width = area[1][0] - area[0][0]
        height = int(VEHICLE_HEIGHT * width / 3.7 / 5)
        height = int(VEHICLE_HEIGHT * width / 20)

        xstart = max(0, area[0][0])
        xstop = min(1279, area[1][0])
        ystop = area[0][1]
        ystart = ystop - height

        # print('baseline: ({:4.0f}, {:4.0f}) - ({:4.0f}, {:4.0f})  <- '.format(xstart, ystart, xstop, ystop), area)
        # cv2.rectangle(draw_img, (xstart, ystart), (xstop, ystop), (255, 0, 0), 1)

        bbox, hog1, hog2, hog3 = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, X_scaler,
                                           hog1, hog2, hog3, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        if bbox:
            bbox_list.extend(bbox)

    return draw_img, bbox_list


def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler,
              hog1, hog2, hog3, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):

    img_tosearch = img[ystart:ystop, xstart:xstop, :]
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
    if hog1 is None or hog2 is None or hog1 is None:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, transform_sqrt=True, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, transform_sqrt=True, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, transform_sqrt=True, feature_vec=False)
    # print('(', xstop - xstart, 'x', ystop - ystart, ') -> ', hog1.shape, hog1.dtype)
    # hog1 = np.zeros((21, 105, 2, 2, 9)).astype(np.float64)
    # hog2 = np.zeros((21, 105, 2, 2, 9)).astype(np.float64)
    # hog3 = np.zeros((21, 105, 2, 2, 9)).astype(np.float64)

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
                bbox.append([[xbox_left, ytop_draw + ystart],
                             [xbox_left + win_draw, ytop_draw + win_draw + ystart]])

    return bbox, hog1, hog2, hog3


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

# Detect Vehicle functions
############################################################


def process_image(image, weight=0.5):

    # 1) Undistort using mtx and dist
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # 2) Create binary image via Combining Threshold
    combined = create_binary_image_adv(undist)
    # return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)  # debug code

    # 3) Perspective Transform
    binary_warped = warper(combined, M)

    # 4) Find Lanes via Sliding Windows: 1st Method

    # 4-1) search lane candidates
    c_left_fit, c_right_fit, out_img = sliding_windows_search(binary_warped)

    # 4-2) Generate x and y values for pixel image
    left_fitx, right_fitx, ploty = fit_quadratic_polynomial(c_left_fit, c_right_fit, binary_warped)

    # 4-3) Check initial status of SlidingWindow function
    left_validity = True
    right_validity = True
    if c_left_fit[1] == 0:
        left_validity = False
    if c_right_fit[1] == 0:
        right_validity = False

    # 5) Determine the lane curvature
    global left_fit, right_fit
    global pre_left_fit, pre_right_fit
    global left_curverad, right_curverad
    global pre_left_curverad, pre_right_curverad

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 575  # meters per pixel in x dimension

    c_left_curverad = measure_curvature(left_fitx, ploty, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
    c_right_curverad = measure_curvature(right_fitx, ploty, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)

    if left_curverad == 0:
        left_curverad = c_left_curverad
    if right_curverad == 0:
        right_curverad = c_right_curverad

    # 6) Sanity Check

    # 6-1) Checking that they have stable coefficients
    c_left_validity, pre_left_fit = sanity_chack_polynomial(c_left_fit, pre_left_fit, thresh0=(-0.0007, 0.0007), thresh1=(-0.5, 0.5), thresh2=(-200, 200))
    c_right_validity, pre_right_fit = sanity_chack_polynomial(c_right_fit, pre_right_fit, thresh0=(-0.0007, 0.0007), thresh1=(-0.5, 0.5), thresh2=(-200, 200))
    if not c_left_validity:
        left_validity = False
    if not c_right_validity:
        right_validity = False

    # 6-2) Checking that they have similar curvature
    c_left_validity, c_right_validity = sanity_chack_curverad(c_left_curverad, c_right_curverad, diff_curverad_thresh=70)
    if not c_left_validity:
        left_validity = False
    if not c_right_validity:
        right_validity = False

    # 6-3) Checking that they are separated by approximately the right distance horizontally
    # 6-4) Checking that they are roughly parallel
    c_validity = sanity_chack_roadwidth(left_fitx, right_fitx, thresh0=(200, 1180), thresh1=(400, 1050), thresh2=(500, 780))  # 640 at horizontal road
    if left_validity and right_validity and not c_validity:
        right_validity = False
        left_validity = False

    # 7) Update Status

    # 7-1) Update Fitting Data
    if c_left_fit[2] != 0 and left_validity:
        left_fit_fifo[0][:] = left_fit_fifo[1][:]
        left_fit_fifo[1][:] = left_fit_fifo[2][:]
        left_fit_fifo[2][:] = left_fit_fifo[3][:]
        left_fit_fifo[3][:] = np.array(c_left_fit)
        left_fit = list((left_fit_fifo[1] + left_fit_fifo[2] + left_fit_fifo[3]) / 3)
    if c_right_fit[2] != 0 and right_validity:
        right_fit_fifo[0][:] = right_fit_fifo[1][:]
        right_fit_fifo[1][:] = right_fit_fifo[2][:]
        right_fit_fifo[2][:] = right_fit_fifo[3][:]
        right_fit_fifo[3][:] = np.array(c_right_fit)
        right_fit = list((right_fit_fifo[1] + right_fit_fifo[2] + right_fit_fifo[3]) / 3)

    # 7-2) Determine Curvature Value
    if left_curverad == 0 or left_validity:
        left_curverad = c_left_curverad
    if right_curverad == 0 or right_validity:
        right_curverad = c_right_curverad

    # 7-3) Detect car position in the lane
    left_fitx, right_fitx, ploty = fit_quadratic_polynomial(left_fit, right_fit, binary_warped)
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    vehicle_offset = 1280 / 2 - lane_center
    vehicle_offset *= xm_per_pix

    # 8) Vehicles Detection
    draw_img = undist
    # undist = find_cars(undist, ystart, ystop, scale, svc, X_scaler,
    #                    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    draw_img, bbox_list = find_cars_multiscale(image, draw_img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # for bbox in bbox_list:
    #     # print('bbox: ', bbox)
    #     cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 1)

    # Update Heatmap
    heatmap_cur = np.zeros_like(image[:, :, 0]).astype(np.uint8)
    add_heat(heatmap_cur, bbox_list)
    heatmap_fifo[1:FRAMENUM, :, :] = heatmap_fifo[0:FRAMENUM - 1, :, :]
    heatmap_fifo[0][:][:] = np.copy(heatmap_cur)
    for f in range(1, FRAMENUM):
        heatmap_cur += heatmap_fifo[f][:][:]
    heatmap_cur = apply_threshold(heatmap_cur, 4)  # 6 or 7
    labelnum, labelimg, contours, centroids = cv2.connectedComponentsWithStats(heatmap_cur)
    # print(' heatmap: ', heatmap.shape)
    # print(' contours: ', contours.shape)

    # Update Car Positions
    # hold_car_positions(bbox_list)


    # X)Drawing

    # X) Overlay Heatmap
    # tmp_heatmap = apply_threshold(heatmap_cur, FRAMENUM + 0)
    # img_heatmap = np.clip(tmp_heatmap, 0, 255)
    # labels = label(img_heatmap)
    # draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)

    # print('labelnum :', labelnum)
    if labelnum > 0:
        for nlabel in range(1, labelnum): 
            x, y, w, h, size = contours[nlabel]
            xg, yg = centroids[nlabel]
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 6)
            # cv2.circle(draw_img, (int(xg), int(yg)), 30, (255, 255, 0), 1)

            # 面積フィルタ
            # if size >= 100 and size <= 1000:
            #     centroid.append([xg, yg, size, curpos])

    # X) Draw mini Heatmap
    px = 10
    py = 90
    font_size = 0.75
    font = cv2.FONT_HERSHEY_DUPLEX

    for f in range(FRAMENUM):
        cv2.putText(draw_img, 'Heatmap {}'.format(f), (px, py - 10), font, font_size, (255, 255, 255))
        mini = np.clip(heatmap_fifo[f] * 16 + 10, 20, 250)
        mini = cv2.resize(mini, (180, 100), interpolation=cv2.INTER_NEAREST)
        mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2RGB)
        draw_img[py:py + mini.shape[0], px:px + mini.shape[1]] = mini
        px += mini.shape[1] + 10

    # X) Draw Detected car positions
    # font_size = 0.5
    # for f in range(FRAMENUM):
    #     cv2.putText(draw_img, 'frame {}'.format(f), (px, py - 10), font, font_size, (255, 255, 255))
    #     posi = car_positions[f]
    #     mini = np.clip(posi * 40 + 20, 20, 240)
    #     mini = cv2.resize(mini, (50, 180), interpolation=cv2.INTER_NEAREST)
    #     mini = cv2.flip(mini, 0)
    #     mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2RGB)
    #     draw_img[py:py + mini.shape[0], px:px + mini.shape[1]] = mini
    #     px += mini.shape[1] + 10




    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    center_fitx = (right_fitx + left_fitx) / 2
    for x, y in zip(center_fitx, ploty):
        cv2.circle(color_warp, (int(x), int(y)), 1, color=[255, 255, 255], thickness=8)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    newwarp = warper(color_warp, Minv)

    # Combine the result with the original image
    font_size = 1.2
    font = cv2.FONT_HERSHEY_DUPLEX
    # infotext = 'offset {:+4.1f}m, curvature left:{:.1f}m,  right:{:.1f}m'.format(vehicle_offset, left_curverad, right_curverad)
    if vehicle_offset < 0:
        infotext = 'car position {:4.2f}m left , curvature {:7.1f}m'.format(-vehicle_offset, (left_curverad + right_curverad) / 2)
    elif vehicle_offset > 0:
        infotext = 'car position {:4.2f}m right, curvature {:7.1f}m'.format(vehicle_offset, (left_curverad + right_curverad) / 2)
    else:
        infotext = 'car position        center, curvature {:7.1f}m'.format((left_curverad + right_curverad)/2)
    cv2.putText(draw_img, infotext, (30, 40), font, font_size, (255, 255, 255))

    return cv2.addWeighted(draw_img, 1, newwarp, 0.3, 0)



######################################
# source and destination points
# Given src and dst points, calculate the perspective transform matrix

perspective_src = np.float32([[585, 460], [695, 460], [1127, 705], [203, 705]])  # adjust
# perspective_src = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial

(width, height) = (1280, 720)
perspective_dst = np.float32([[320, 0], [width - 320, 0], [width - 320, height - 0], [320, height - 0]])

# Calculate the Perspective Transformation Matrix and its invert Matrix
M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
Minv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)

######################################
# set Perspective Matrix

set_perspective_matrix()
######################################
# output to video files

left_fit = [0, 0, 360]
right_fit = [0, 0, 920]
left_fit_fifo = np.array([left_fit, left_fit, left_fit, left_fit]).astype(np.float)
right_fit_fifo = np.array([right_fit, right_fit, right_fit, right_fit]).astype(np.float)
pre_left_fit = [0, 0, 0]
pre_right_fit = [0, 0, 0]

left_curverad = 0
right_curverad = 0
pre_left_curverad = 0
pre_right_curverad = 0
car_positions = np.zeros((FRAMENUM, DISTANCE_NUM, LANE_NUM), dtype=np.uint8)
heatmap_fifo = np.zeros((FRAMENUM, 720, 1280), dtype=np.uint8)

white_output = './test_video_out.mp4'
clip1 = VideoFileClip('../test_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

left_fit = [0, 0, 360]
right_fit = [0, 0, 920]
left_fit_fifo = np.array([left_fit, left_fit, left_fit, left_fit]).astype(np.float)
right_fit_fifo = np.array([right_fit, right_fit, right_fit, right_fit]).astype(np.float)
pre_left_fit = [0, 0, 0]
pre_right_fit = [0, 0, 0]

left_curverad = 0
right_curverad = 0
pre_left_curverad = 0
pre_right_curverad = 0
car_positions = np.zeros((FRAMENUM, DISTANCE_NUM, LANE_NUM), dtype=np.uint8)
heatmap_fifo = np.zeros((FRAMENUM, 720, 1280), dtype=np.uint8)

white_output = './project_video_out.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

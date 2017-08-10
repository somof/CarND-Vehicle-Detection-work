
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle

from functions_vehicle import *
from functions_lane import *


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  #16    # Number of histogram bins
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
    undist = find_cars(undist, ystart, ystop, scale, svc, X_scaler,
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


    # X)Drawing

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
    cv2.putText(undist, infotext, (30, 50), font, font_size, (255, 255, 255))

    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)



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

white_output = './project_video_out.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

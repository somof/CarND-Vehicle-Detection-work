
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle

from lesson_functions34 import *


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, sobel_thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary gray of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the absolute value of the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # apply a threshold, and create a binary image result
    binary_output = np.zeros(absgraddir.shape).astype(np.uint8)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def hls_select(image, hthresh=(0, 255), sthresh=(0, 255), ithresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to each channel
    h_channel = hls[:, :, 0]
    i_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(((h_channel >= hthresh[0]) & (h_channel <= hthresh[1])) &
                   ((s_channel >= sthresh[0]) & (s_channel <= sthresh[1])) &
                   ((i_channel >= ithresh[0]) & (i_channel <= ithresh[1])))] = 1
    return binary_output


def rgb_select(image, rthresh=(0, 255), gthresh=(0, 255), bthresh=(0, 255)):
    b_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    r_channel = image[:, :, 2]
    binary_output = np.zeros_like(g_channel)
    binary_output[(((r_channel >= rthresh[0]) & (r_channel <= rthresh[1])) &
                   ((g_channel >= gthresh[0]) & (g_channel <= gthresh[1])) &
                   ((b_channel >= bthresh[0]) & (b_channel <= bthresh[1])))] = 1
    return binary_output


def warper(img, M):
    # Compute and apply perpective transform
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped


def create_binary_image_light(image):
    """
    output a binary image that show lane lines
    light algorithm
    """

    # 1) White Line
    rgb_white = rgb_select(image, rthresh=(200, 255), gthresh=(200, 255), bthresh=(200, 255))  # white line
    rgb_excess = rgb_select(image, rthresh=(250, 255), gthresh=(250, 255), bthresh=(250, 255))  # white line

    # 2) Yellow Line
    hls_yellow1 = hls_select(image, hthresh=(10, 30), ithresh=(50, 150), sthresh=(30, 255))  # yellow line dark
    hls_yellow2 = hls_select(image, hthresh=(20, 30), ithresh=(120, 250), sthresh=(30, 255))  # yellow line light

    # combined = np.zeros_like(dir_binary)
    combined = np.zeros((rgb_white.shape), dtype=np.uint8)
    combined[((hls_yellow1 == 1) | (hls_yellow2 == 1))] = 1  # yellow line
    combined[((rgb_white == 1) & (rgb_excess != 1))] = 1  # White line

    return combined


def create_binary_image_adv(image):
    """
    output a binary image that show lane lines
    """

    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(image, (200, 200, 200), (255, 255, 255))

    combined = yellow | white | white_2 | white_3
    return combined



def create_binary_image(image):
    """
    output a binary image that show lane lines
    """
    # 1) Line Edge
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, sobel_thresh=(25, 50))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, sobel_thresh=(50, 150))
    mag_binary = mag_thresh(gray, sobel_kernel=15, mag_thresh=(50, 250))
    dir_binary = dir_threshold(gray, sobel_kernel=7, thresh=(0.7, 1.3))
    # hls_binary = hls_select(image, hthresh=(50, 100), ithresh=(0, 255), sthresh=(90, 190))  # Asphalt color
    hls_binary = hls_select(image, hthresh=(0, 255), ithresh=(0, 255), sthresh=(90, 190))  # Asphalt color

    # 3) Concrete
    hls_binary2 = hls_select(image, hthresh=(50, 100), ithresh=(0, 255), sthresh=(90, 190))  # shadow

    # combined = np.zeros_like(dir_binary)
    combined = np.zeros((mag_binary.shape), dtype=np.uint8)
    combined[((gradx == 1) | (grady == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)) & (hls_binary2 != 1)] = 1

    return combined


def get_base_position(image, pos='left'):
    """
    input:image must have binary values
    """
    # Get two base positions from the histgram
    if pos == 'left':
        base = 360
    else:
        base = 920

    # for div in (10, 9, 8, 7, 6, 5, 4, 3, 2):
    #     # Take a histogram of the bottom half of the image
    #     histogram = np.sum(image[image.shape[0] // div:, :], axis=0)
    #     # Find the peak of the left and right halves of the histogram
    #     # These will be the starting point for the left and right lines
    #     midpoint = np.int(histogram.shape[0] / 2)
    #     if pos == 'left':
    #         if 30 < np.max(histogram[:midpoint]):
    #             print('left base :', np.argmax(histogram[:midpoint]), '  with ', div, '/', midpoint)
    #             return np.argmax(histogram[:midpoint])
    #     else:
    #         if 30 < np.max(histogram[midpoint:]):
    #             # print('right base:', np.argmax(histogram[midpoint:]) + midpoint, '  with ', div, '/', midpoint)
    #             return np.argmax(histogram[midpoint:]) + midpoint

    return base


def sliding_windows_search(image):
    """
    input:image must have binary values
    """

    # Get two base positions from the histgram
    leftx_base = get_base_position(image, 'left')
    rightx_base = get_base_position(image, 'right')

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_different = 0
    rightx_different = 0
    # Set Searching Parameters
    margin = 90  # Set the width of the windows +/- margin
    minpix = 70  # Set minimum number of pixels found to recenter window
    # Create empty lists to receive left and right lane pixel indices
    left_lane_index = []
    right_lane_index = []

    # Step through the windows one by one
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        if window == 0:
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height + window_height // 2
            win_xleft_low = leftx_current - int(margin * 1.5)
            win_xleft_high = leftx_current + int(margin * 1.5)
            win_xright_low = rightx_current - int(margin * 1.5)
            win_xright_high = rightx_current + int(margin * 1.5)
        else:
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height + window_height // 2
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_index = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_index = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_index.append(good_left_index)
        right_lane_index.append(good_right_index)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_index) > minpix:
            leftx_candidate = np.int(np.mean(nonzerox[good_left_index]))
            leftx_different = leftx_candidate - leftx_current
            leftx_current = leftx_candidate
        else:
            leftx_current += leftx_different + leftx_different
        if len(good_right_index) > minpix:
            rightx_candidate = np.int(np.mean(nonzerox[good_right_index]))
            rightx_different = rightx_candidate - rightx_current
            rightx_current = rightx_candidate
        else:
            rightx_current += rightx_different + rightx_different

    # Concatenate the arrays of indices
    left_lane_index = np.concatenate(left_lane_index)
    right_lane_index = np.concatenate(right_lane_index)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_index]
    lefty = nonzeroy[left_lane_index]
    rightx = nonzerox[right_lane_index]
    righty = nonzeroy[right_lane_index]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except TypeError:
        left_fit = [0, 0, 0]

    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        right_fit = [0, 0, 0]

    out_img[nonzeroy[left_lane_index], nonzerox[left_lane_index]] = [255, 150, 150]
    out_img[nonzeroy[right_lane_index], nonzerox[right_lane_index]] = [150, 150, 255]

    return left_fit, right_fit, out_img


def measure_curvature(xs, ys, ym_per_pix, xm_per_pix):
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ys * ym_per_pix, xs * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ys)
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad


def fit_quadratic_polynomial(c_left_fit, c_right_fit, image):
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = c_left_fit[0] * ploty**2 + c_left_fit[1] * ploty + c_left_fit[2]
    right_fitx = c_right_fit[0] * ploty**2 + c_right_fit[1] * ploty + c_right_fit[2]

    return left_fitx, right_fitx, ploty


def sanity_chack_polynomial(fit, pre_fit, thresh0=(-0.0007, 0.0007), thresh1=(-0.5, 0.5), thresh2=(-200, 200)):
    if fit[1] == 0:
        return False, pre_fit
    if pre_fit[1] == 0:
        return True, fit

    diff = pre_fit - fit

    if diff[0] < thresh0[0] or thresh0[1] < diff[0]:
        return False, fit
    if diff[1] < thresh1[0] or thresh1[1] < diff[1]:
        return False, fit
    if diff[2] < thresh2[0] or thresh2[1] < diff[2]:
        return False, fit

    return True, fit


def sanity_chack_curverad(left_curverad, right_curverad, diff_curverad_thresh=40):
    global pre_left_curverad, pre_right_curverad
    if pre_left_curverad == 0 or pre_right_curverad == 0:
        pre_left_curverad = left_curverad
        pre_right_curverad = right_curverad
        return True, True

    diff_left_curverad = 100 * abs(pre_left_curverad - left_curverad) / pre_left_curverad
    diff_right_curverad = 100 * abs(pre_right_curverad - right_curverad) / pre_right_curverad

    left_validity = True
    right_validity = True

    if diff_curverad_thresh < diff_left_curverad:
        left_validity = False
    if diff_curverad_thresh < diff_right_curverad:
        right_validity = False

    pre_left_curverad = left_curverad
    pre_right_curverad = right_curverad

    return left_validity, right_validity


def sanity_chack_roadwidth(left_fitx, right_fitx, thresh0=(300, 1200), thresh1=(300, 1200), thresh2=(300, 1200)):

    road_width0 = right_fitx[0] - left_fitx[0]
    road_width1 = right_fitx[len(right_fitx)//2] - left_fitx[len(left_fitx)//2]
    road_width2 = right_fitx[-1] - left_fitx[-1]

    if road_width0 < thresh0[0] or thresh0[1] < road_width0:
        return False
    if road_width1 < thresh1[0] or thresh1[1] < road_width1:
        return False
    if road_width2 < thresh2[0] or thresh2[1] < road_width2:
        return False

    return True


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
    windows = slide_window(undist, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(undist, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)












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

    # Draw Vehicles BBox
    undist = draw_boxes(undist, hot_windows, color=(0, 0, 255), thick=6)

    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


######################################
# source and destination points
# Given src and dst points, calculate the perspective transform matrix

# perspective_src = np.float32([[631, 425], [649, 425], [1055, 675], [265, 675]])  # trial
# perspective_src = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 720], [203, 720]])  # sample data
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 700], [203, 700]])  # ignore bonnet
# perspective_src = np.float32([[582, 460], [698, 460], [1127, 695], [203, 695]])  # a little adjustment
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 695], [203, 695]])  # a little adjustment
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 685], [203, 685]])  # prevent bonnnet
perspective_src = np.float32([[585, 460], [695, 460], [1127, 705], [203, 705]])  # adjust
# perspective_src = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial

(width, height) = (1280, 720)
perspective_dst = np.float32([[320, 0], [width - 320, 0], [width - 320, height - 0], [320, height - 0]])

# Calculate the Perspective Transformation Matrix and its invert Matrix
M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
Minv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)

######################################
# process frame by frame for developing

trapezoid = []
trapezoid.append([[perspective_src[0][0], perspective_src[0][1], perspective_src[1][0], perspective_src[1][1]]])
trapezoid.append([[perspective_src[1][0], perspective_src[1][1], perspective_src[2][0], perspective_src[2][1]]])
trapezoid.append([[perspective_src[2][0], perspective_src[2][1], perspective_src[3][0], perspective_src[3][1]]])
trapezoid.append([[perspective_src[3][0], perspective_src[3][1], perspective_src[0][0], perspective_src[0][1]]])

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

print('svc: ', svc)
print('X_scaler: ', X_scaler)
print('orient: ', orient)
print('pix_per_cell: ', pix_per_cell)
print('cell_per_block: ', cell_per_block)
print('spatial_size: ', spatial_size)
print('hist_bins: ', hist_bins)




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

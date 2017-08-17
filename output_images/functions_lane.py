import numpy as np
import cv2

left_curverad = 0
right_curverad = 0
pre_left_curverad = 0
pre_right_curverad = 0


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
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
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
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary gray of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
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
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
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
    road_width1 = right_fitx[len(right_fitx) // 2] - left_fitx[len(left_fitx) // 2]
    road_width2 = right_fitx[-1] - left_fitx[-1]

    if road_width0 < thresh0[0] or thresh0[1] < road_width0:
        return False
    if road_width1 < thresh1[0] or thresh1[1] < road_width1:
        return False
    if road_width2 < thresh2[0] or thresh2[1] < road_width2:
        return False

    return True

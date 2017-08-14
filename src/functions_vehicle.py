import numpy as np
import cv2
from functions_training import convert_color
from functions_training import get_hog_features
from functions_training import bin_spatial
from functions_training import color_hist

M = []
Minv = []
search_area = []


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
    for y in range(5, 24, 1):
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

    bbox_list = []
    for area in search_area:
        scale = 1.5

        width = area[1][0] - area[0][0]
        height = int(1.75 * width / 20)

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

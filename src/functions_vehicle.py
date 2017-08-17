import cv2
import numpy as np
# from scipy.ndimage.measurements import label
from functions_training import convert_color
from functions_training import get_hog_features
from functions_training import bin_spatial
from functions_training import color_hist


VIEW_WIDTH     = 3.7 * 5
VEHICLE_HEIGHT = 1.5  # 1.65  # meter
LANENUM        =  5
FRAMENUM       =  5  # FRAMENO_0 is the current frame

heatmap_fifo = np.zeros((FRAMENUM, 720, 1280), dtype=np.uint8)

# distance_map = range(6, 19, 2)
distance_map = (6.6, 7.2, 8, 9, 10.5, 13, 18)


def set_perspective_matrix():

    global M2, M2inv, search_area

    perspective_2d = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial
    perspective_3d = np.float32([[-1.85, 40], [1.85, 40], [1.85, 5], [-1.85, 5]])

    M2 = cv2.getPerspectiveTransform(perspective_3d, perspective_2d)
    M2inv = cv2.getPerspectiveTransform(perspective_2d, perspective_3d)

    print('Search Area:')
    search_area = []
    for y in distance_map:
        x = - VIEW_WIDTH / 2
        x0 = (M2[0][0] * x + M2[0][1] * y + M2[0][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        y0 = (M2[1][0] * x + M2[1][1] * y + M2[1][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        #
        x = VIEW_WIDTH / 2
        x1 = (M2[0][0] * x + M2[0][1] * y + M2[0][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        y1 = (M2[1][0] * x + M2[1][1] * y + M2[1][2]) / (M2[2][0] * x + M2[2][1] * y + M2[2][2])
        #
        search_area.append([[int(x0), int(y0)], [int(x1), int(y1)]])
        print(' {:4.1f} : ({:+8.1f},{:+8.1f}) - ({:+8.1f},{:+8.1f})'.format(y, x0, y0, x1, y1))


def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler,
              transform_sqrt, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    # img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale),
                                      np.int(imshape[0] / scale)))

    # print('   shape :', ctrans_tosearch.shape)
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    # nfeat_per_block = orient * cell_per_block**2
    # print('  nyblocks: ', nyblocks)

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    nysteps = max(1, nysteps)
    # print('  nyblocks: ', nyblocks)

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, transform_sqrt=transform_sqrt, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, transform_sqrt=transform_sqrt, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, transform_sqrt=transform_sqrt, feature_vec=False)
    # print('(', xstop - xstart, 'x', ystop - ystart, ') -> ', hog1.shape, hog1.dtype)

    if 1 < nysteps:
        print('    step : {} x {}'.format(nxsteps, nysteps))

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
                xbox_left = np.int(xleft * scale + xstart)  # add offset
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox.append([[xbox_left, ytop_draw + ystart],
                             [xbox_left + win_draw, ytop_draw + win_draw + ystart]])

    return bbox


def find_cars_multiscale(image, svc, X_scaler,
                         transform_sqrt, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    global search_area

    bbox_list = []
    for area in search_area:
        # print('  area ({:+8.1f},{:+8.1f}) - ({:+8.1f},{:+8.1f})'.format(area[0][0], area[0][1], area[1][0], area[1][1]))

        width = area[1][0] - area[0][0]
        height = int(VEHICLE_HEIGHT * width / VIEW_WIDTH)
        scale = height / 64.0

        xstart = max(area[0][0], 0)
        xstop = min(area[1][0], 1279)
        ystop = area[0][1]
        ystart = ystop - height

        print('  area: ({:4.0f}, {:4.0f}) - ({:4.0f}, {:4.0f}) '.format(xstart, ystart, xstop, ystop), end='')
        # print('  <- ', area, end='')
        print('  scale: {:4.2f}'.format(scale))
        bbox = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, X_scaler,
                         transform_sqrt, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if bbox:
            bbox_list.extend(bbox)

    return bbox_list


def overlay_search_area(draw_img):

    global search_area

    for area in search_area:

        width = area[1][0] - area[0][0]
        height = int(VEHICLE_HEIGHT * width / VIEW_WIDTH)

        xstart = max(0, area[0][0])
        xstop = min(1279, area[1][0])
        ystop = area[0][1]
        ystart = ystop - height

        # print('baseline: ({:4.0f}, {:4.0f}) - ({:4.0f}, {:4.0f})  <- '.format(xstart, ystart, xstop, ystop), area)
        cv2.rectangle(draw_img, (xstart, ystart), (xstop, ystop), (255, 0, 0), 1)

    return draw_img


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


def reset_hetmap_fifo():
    global heatmap_fifo

    heatmap_fifo = np.zeros((FRAMENUM, 720, 1280), dtype=np.uint8)


def select_bbox_with_heatmap(image, bbox_list, threshold=4):

    global heatmap_fifo

    heatmap_cur = np.zeros_like(image[:, :, 0]).astype(np.uint8)
    add_heat(heatmap_cur, bbox_list)
    # heatmap_cur = apply_threshold(heatmap_cur, 1)

    heatmap_fifo[1:FRAMENUM, :, :] = heatmap_fifo[0:FRAMENUM - 1, :, :]
    heatmap_fifo[0][:][:] = np.copy(heatmap_cur)

    for f in range(1, FRAMENUM):
        heatmap_cur += heatmap_fifo[f][:][:]

    heatmap_cur = apply_threshold(heatmap_cur, threshold)
    labelnum, labelimg, contours, centroids = cv2.connectedComponentsWithStats(heatmap_cur)

    return labelnum, contours


def overlay_heatmap_fifo(draw_img, px=10, py=90, size=(180, 100)):

    font_size = 0.5
    font = cv2.FONT_HERSHEY_DUPLEX
    font = cv2.FONT_HERSHEY_COMPLEX

    for f in range(FRAMENUM):
        cv2.putText(draw_img, 'Heatmap {}'.format(f), (px, py - 10), font, font_size, (255, 255, 255))
        mini = np.clip(heatmap_fifo[f] * 4 + 10, 20, 255)
        mini = cv2.resize(mini, size, interpolation=cv2.INTER_NEAREST)
        mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2RGB)
        draw_img[py:py + mini.shape[0], px:px + mini.shape[1]] = mini
        px += mini.shape[1] + 10

    return draw_img

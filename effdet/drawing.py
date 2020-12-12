from skimage.draw import rectangle_perimeter, set_color

import cv2
import numpy as np

# Yaak Yetturbium color


def draw_rectangle(frame, boxes, color=(108, 233, 132)):

    h, w = frame.shape[:2]

    for b in boxes:
        b = list(map(int, b))
        # x_min, y_min, x_max, y_max
        # https://scikit-image.org/docs/dev/api/skimage.draw.html#rectangle-perimeter
        start = (b[1], b[0])
        end = (b[3], b[2])
        try:
            rr, cc = rectangle_perimeter(start, end=end, shape=(h, w), clip=True)
            set_color(
                frame,
                (rr, cc),
                color=color,
            )
        except Exception as err:
            continue

    return frame


def pixelate(image, blocks=3):

    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    # return the pixelated blurred image
    return image


def redact_regions(image, detections):

    (h, w) = image.shape[:2]

    for label, confidence, bbox in detections:
        (left, top, right, bottom) = bbox
        if right > w or bottom > h:
            continue
        redact_cover = image[top:bottom, left:right, :]
        cover = pixelate(redact_cover, blocks=4)
        image[top:bottom, left:right, :] = cover

    return image

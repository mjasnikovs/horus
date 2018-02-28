import cv2 as cv
import numpy as np
import argparse
import math

from helper import downScaleImage, RGB, createMask, contourPixels
from helper import midpoint

from imutils import perspective
from scipy.spatial import distance
from statistics import median

frame = cv.imread('./TEMP/picture0.jpg')

# crop, to area of interest
height, width = frame.shape[:2]
frame = frame[400:1300, 0:width]
height, width = frame.shape[:2]
center = (width / 2, height / 2)

frame, mask = createMask(
    frame,
    x=int(width / 2 - 150),
    y=int(height - 100),
    w=300,
    h=50,
    tolerance=15,
    draw=True
)

# find contours and extract biggest
_, cnts, hier = cv.findContours(
    mask.copy(),
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

if len(cnts):
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    c = cnts[0]

    # box area/points/angle
    area = cv.contourArea(c)
    rect = cv.minAreaRect(c)
    angle = rect[2]
    box = cv.boxPoints(rect)
    tl, tr, br, bl = perspective.order_points(box)

    # draw detected line
    cv.drawContours(frame, [box.astype('int')], -1, RGB.Lime, 2)

    # rotate thresh image to contour 90 degree of angle
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    mask = cv.warpAffine(mask, rotation_matrix, (width, height))

    x, y, w, h = cv.boundingRect(c)
    crop = mask[y:y + w, x: x + w]
    ch, cw = frame.shape[:2]

    heightCrop = crop[0:ch, int(cw / 2 - 25):int(cw / 2 + 25)]
    _, heightPix, targ = contourPixels(heightCrop)
    top, _, bottom = targ[:3]

    if heightPix >= 40:
        widthCrop = crop[
            int(top[1] + heightPix - 40):int(bottom[1]),
            0:cw
        ]
        widthPix, _, _ = contourPixels(widthCrop)[:1]
    else:
        widthPix = None

    print('heightPix', heightPix)
    print('widthPix', widthPix)


images = np.vstack((
    frame,
    cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
))

cv.imshow('images', downScaleImage(images, 3))
cv.waitKey(0) & 0xFF
cv.destroyAllWindows()

import cv2
import math
import time
import numpy as np
import argparse
import imutils

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

# config
scale = 2

frameWidth = 2304
frameHeight = 1536
frameCroopY = [250, 550]

pixelsPerMetricX = 1.050631579
pixelsPerMetricY = 272 / 38.58448571

windowsName = 'Window Name'

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def processFrame(image):
    # detect ratio
    ratio = image.shape[0] / float(image.shape[0])

    mutate = image[frameCroopY[0]:frameCroopY[1], 0:frameWidth]
    mutate = cv2.cvtColor(mutate, cv2.COLOR_RGB2GRAY)

    mutate = cv2.bilateralFilter(mutate, 50,  50, cv2.BORDER_WRAP)
    # mutate = cv2.Canny(mutate, 4, 255)
    mutate = cv2.dilate(mutate, None, iterations=1)
    mutate = cv2.erode(mutate, None, iterations=1)

    treshold = cv2.threshold(mutate, 60, 255, cv2.THRESH_BINARY)[1]
    _, cnts, _ = cv2.findContours(
        treshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if cv2.contourArea(c) > 100000:
            if len(approx) == 4:
                    # compute the rotated bounding box of the contour
                    orig = image.copy()
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(
                        box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")

                    # order the points in the contour such that they appear
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order, then draw the outline of the rotated bounding
                    # box
                    box = perspective.order_points(box)
                    # cv2.drawContours(
                    #   image, [box.astype("int")], -1, (0, 255, 0), 2)

                    # loop over the original points and draw them
                    # for (x, y) in box:
                    #     cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

                    # unpack the ordered bounding box, then compute the midpoint
                    # between the top-left and top-right coordinates, followed by
                    # the midpoint between bottom-left and bottom-right coordinates
                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)

                    # compute the midpoint between the top-left and top-right points,
                    # followed by the midpoint between the top-righ and bottom-right
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)

                    # draw the midpoints on the image
                    # cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                    # cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                    # cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                    # cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                    # draw lines between the midpoints
                    # cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    #          (255, 0, 255), 2)
                    # cv2.line(image, (int(tlblX), frameCroopY[0] + int(tlblY)), (int(trbrX), frameCroopY[0] + int(trbrY)),
                    #         (0, 255, 0), 50)

                    cv2.rectangle(
                        image,
                        (int(tlblX), frameCroopY[0] + int(tlblY)),
                        (int(trbrX), frameCroopY[1] + int(trbrY)),
                        (92, 66, 244),
                        -1)

                    # compute the Euclidean distance between the midpoints
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                    # compute the size of the object
                    dimA = dA / pixelsPerMetricX
                    dimB = dB / pixelsPerMetricY

                    # draw the object sizes on the image
                    # cv2.putText(image, "{:.1f} mm".format(dimA),
                    #             (int(tltrX - 15), int(tltrY - 10)
                    #              ), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.65, (255, 255, 255), 2)
                    cv2.putText(image, "{:.1f} mm".format(dimB),
                                (int(tlblX), frameCroopY[1] + int(trbrY) - 10
                                ), cv2.FONT_HERSHEY_SIMPLEX,
                                4, (255, 255, 255), 10)
    # resize
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width / scale), int(height / scale)))

def playvideo():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

    while(True):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print('release')
            break

        frame = processFrame(frame)

        cv2.namedWindow(windowsName)
        cv2.startWindowThread()
        cv2.imshow(windowsName, frame)

        k = cv2.waitKey(1)

        if k == 27:
            break

    cv2.destroyAllWindows()

playvideo()

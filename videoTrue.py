import cv2
import math
import time
import numpy as np
import argparse
import imutils

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

#https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv

pixelsPerMetricX = 1.050631579
pixelsPerMetricY = 1.036231884

scale = 2

frameWidth = 2304
frameHeight = 1536
frameCroopY = [650, 950]

windowsName = 'Window Name'

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

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


def processFrame(frame):
    frame = frame[frameCroopY[0]:frameCroopY[1], 0:frameWidth]
    liveFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    liveFrame = cv2.bilateralFilter(liveFrame, 10, 50, cv2.BORDER_WRAP)
    liveFrame = cv2.Canny(liveFrame, 65, 130)
    liveFrame = cv2.dilate(liveFrame, None, iterations=1)
    liveFrame = cv2.erode(liveFrame, None, iterations=1)

    # liveFrame = cv2.bilateralFilter(liveFrame, 10, 50, cv2.BORDER_WRAP)
    # liveFrame = cv2.Canny(liveFrame, 70, 140, 1)

    _, cnts, _ = cv2.findContours(
        liveFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts):
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        for c in cnts:
            if cv2.contourArea(c) > 100:
                # compute the rotated bounding box of the contour
                orig = frame.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

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
                cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # draw lines between the midpoints
                cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
                cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # compute the size of the object
                dimA = dA / pixelsPerMetricX
                dimB = dB / pixelsPerMetricY

                # draw the object sizes on the image
                cv2.putText(frame, "{:.1f} mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
                cv2.putText(frame, "{:.1f} mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    liveFrame = cv2.cvtColor(liveFrame, cv2.COLOR_GRAY2BGR)
    combined = np.vstack((liveFrame, frame))

    height, width = combined.shape[:2]
    return cv2.resize(combined, (int(width / scale), int(height / scale)))


playvideo()

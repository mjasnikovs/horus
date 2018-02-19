import cv2 as cv
import numpy as np
import argparse
import statistics

from helper import webcamStream, downScaleImage, RGB
from imutils import contours, perspective
from scipy.spatial import distance

parser = argparse.ArgumentParser()

parser.add_argument('--downscale', type=int,
                    help='Downscale display image size')

parser.add_argument('--width', type=int, help='Camera resolution - width')
parser.add_argument('--height', type=int, help='Camera resolution - height')
parser.add_argument('--fps', type=int, help='Target camera FPS')
parser.add_argument('--opencl', type=int, help='Use opencl')
parser.add_argument('--name', type=str, help='Camera name')

args = parser.parse_args()

WINDOWS_NAME = 'camera reader example'
SCALE_ARGS = 1 if (args.downscale is None) else args.downscale
WIDTH_ARGS = 4096 if (args.width is None) else args.width
HEIGHT_ARGS = 2160 if (args.height is None) else args.height
FPS_ARGS = 5 if (args.fps is None) else args.fps
NAME_ARGS = 'camera' if (args.name is None) else args.name

# config
frameCroopY = [650, 1250]
hsvCS = 55

# width
pixelsPerMetricX = 3.849631579
# length
pixelsPerMetricY = 3.913043478


def colorRange(B, G, R, range):
    return np.array([[B + range, G + range, R + range]])


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def processFrame(frame):
    # croop, to area of interest
    height, width = frame.shape[:2]
    frame = frame[frameCroopY[0]:frameCroopY[1], 0:width]
    height, width = frame.shape[:2]
    centerY, centerX = (int(height / 2), int(width / 2))

    frame = cv.bilateralFilter(frame, 10, 60, cv.BORDER_WRAP)
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    B1, G1, R1 = hsvFrame[centerY, centerX]
    B2, G2, R2 = hsvFrame[centerY + 50, centerX]
    B3, G3, R3 = hsvFrame[centerY - 50, centerX]
    B4, G4, R4 = hsvFrame[centerY, centerX + 50]
    B5, G5, R5 = hsvFrame[centerY, centerX - 50]

    B = statistics.median([B1, B2, B3, B4, B5])
    G = statistics.median([G1, G2, G3, G4, G5])
    R = statistics.median([R1, R2, R3, R4, R5])

    lowerRange = colorRange(B, G, R, -hsvCS)
    upperRange = colorRange(B, G, R, hsvCS)

    mask = cv.inRange(hsvFrame, lowerRange, upperRange)

    cv.circle(frame, (centerY, centerX), 5, RGB.Red, -1)
    cv.circle(frame, (centerY + 50, centerX), 5, RGB.Red, -1)
    cv.circle(frame, (centerY - 50, centerX), 5, RGB.Red, -1)
    cv.circle(frame, (centerY, centerX + 50), 5, RGB.Red, -1)
    cv.circle(frame, (centerY, centerX - 50), 5, RGB.Red, -1)

    _, cnts, _ = cv.findContours(
        mask.copy(),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if len(cnts):
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        for c in cnts:
            if cv.contourArea(c) > 10000:
                # compute the rotated bounding box of the contour
                # orig = frame.copy()
                box = cv.minAreaRect(c)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                cv.drawContours(frame, [box.astype("int")], -1, RGB.Cyan, 2)

                for (x, y) in box:
                    cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the
                # top-left and top-right points
                # followed by the midpoint between the
                # top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                cv.circle(frame, (int(tltrX), int(tltrY)), 5, RGB.Red, -1)
                cv.circle(frame, (int(blbrX), int(blbrY)), 5, RGB.Red, -1)
                cv.circle(frame, (int(tlblX), int(tlblY)), 5, RGB.Red, -1)
                cv.circle(frame, (int(trbrX), int(trbrY)), 5, RGB.Red, -1)

                # draw lines between the midpoints
                cv.line(frame, (int(tltrX), int(tltrY)),
                        (int(blbrX), int(blbrY)),
                        RGB.Magenta, 2)
                cv.line(frame, (int(tlblX), int(tlblY)),
                        (int(trbrX), int(trbrY)),
                        RGB.Magenta, 2)

                # compute the Euclidean distance between the midpoints
                dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # compute the size of the object
                dimA = dA / pixelsPerMetricX
                dimB = dB / pixelsPerMetricY

                # draw the object sizes on the image
                cv.putText(frame, "{:.1f} mm".format(dimA),
                           (int(tltrX - 15), int(tltrY - 10)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           2, RGB.Lime, 2)
                cv.putText(frame, "{:.1f} mm".format(dimB),
                           (int(trbrX + 10), int(trbrY)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           2, RGB.Lime, 2)

    return frame


MATRIX = np.loadtxt('calibrations/' + NAME_ARGS + '_matrix.txt', delimiter=',')
DISTORTION = np.loadtxt('calibrations/' + NAME_ARGS + '_distortion.txt',
                        delimiter=',')

stream = webcamStream(1, WIDTH_ARGS, HEIGHT_ARGS, FPS_ARGS).start()
stream.calibrateCamera(MATRIX, DISTORTION)
stream.stream.set(cv.CAP_PROP_EXPOSURE, -7)

cv.namedWindow(WINDOWS_NAME)
cv.startWindowThread()

while True:
    frame = stream.readClean()

    if frame is not None:
        frame = processFrame(frame)

        if (SCALE_ARGS > 1):
            frame = downScaleImage(frame, SCALE_ARGS)

        frame = cv.putText(frame, 'FPS:{:3d}'.format(stream.readFPS()),
                           (20, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)

        cv.imshow(WINDOWS_NAME, frame)
        k = cv.waitKey(100)

        if k == 27:
            break

cv.destroyAllWindows()
stream.stop()

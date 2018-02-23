import cv2 as cv
import numpy as np
import argparse

from helper import webcamStream, downScaleImage, scanColorRangeMedian, RGB
from helper import midpoint

from imutils import perspective
from scipy.spatial import distance
from statistics import median

parser = argparse.ArgumentParser()

parser.add_argument('--downscale', type=int,
                    help='Downscale display image size')

parser.add_argument('--width', type=int, help='Camera resolution - width')
parser.add_argument('--height', type=int, help='Camera resolution - height')
parser.add_argument('--fps', type=int, help='Target camera FPS')
parser.add_argument('--opencl', type=int, help='Use opencl')
parser.add_argument('--name', type=str, help='Camera name')

args = parser.parse_args()

WINDOWS_NAME = 'A4 test'
SCALE_ARGS = 1 if (args.downscale is None) else args.downscale
WIDTH_ARGS = 4096 if (args.width is None) else args.width
HEIGHT_ARGS = 2160 if (args.height is None) else args.height
FPS_ARGS = 5 if (args.fps is None) else args.fps
NAME_ARGS = None if (args.name is None) else args.name

stream = webcamStream(1, WIDTH_ARGS, HEIGHT_ARGS, FPS_ARGS).start()

if NAME_ARGS is not None:
    MATRIX = np.loadtxt('calibrations/' + NAME_ARGS + '_matrix.txt',
                        delimiter=',')
    DISTORTION = np.loadtxt('calibrations/' + NAME_ARGS + '_distortion.txt',
                            delimiter=',')
    stream.calibrateCamera(MATRIX, DISTORTION)

widthPixArray = list()
lengthPixArray = list()


def rectangleSize(mask, frame):
    global widthPixArray
    frame = frame.copy()

    _, cnts, hier = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if len(cnts):
        # cnts, _ = contours.sort_contours(cnts, method='left-to-right')

        for c in cnts:
            area = cv.contourArea(c)
            if area > 10000:
                rect = cv.minAreaRect(c)
                box = cv.boxPoints(rect)
                tl, tr, br, bl = perspective.order_points(box)

                # draw 4 points/ordered
                cv.circle(frame, tuple(tl), 2, RGB.Lime, -1)
                cv.circle(frame, tuple(tr), 2, RGB.Lime, -1)
                cv.circle(frame, tuple(br), 2, RGB.Lime, -1)
                cv.circle(frame, tuple(bl), 2, RGB.Lime, -1)

                # > Width measurement
                # extract width center points
                topWx, topWy = midpoint(tl, tr)
                botWx, botWy = midpoint(bl, br)

                cv.circle(frame, (int(topWx), int(topWy)), 5, RGB.Purple, -1)
                cv.circle(frame, (int(botWx), int(botWy)), 5, RGB.Purple, -1)

                widthPix = distance.euclidean((topWx, topWy), (botWx, botWy))
                widthPixArray.append(widthPix)

                cv.putText(frame, 'Wpix: {:.5f}'.format(median(widthPixArray)),
                           (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX,
                           1, RGB.Black, 2)

                msg = '{:4d}.00 mm'.format(round(widthPix / (333.66 / 210)))
                cv.putText(frame, msg,
                           (int(topWx), int(topWy)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           1.5, RGB.Lime, 2)

                # > length measurement
                # extract lenght center points
                topLx, topLy = midpoint(tl, bl)
                botLx, botLy = midpoint(tr, br)

                cv.circle(frame, (int(topLx), int(topLy)), 5, RGB.Blue, -1)
                cv.circle(frame, (int(botLx), int(botLy)), 5, RGB.Blue, -1)

                lengthPix = distance.euclidean((topLx, topLy), (botLx, botLy))
                lengthPixArray.append(lengthPix)

                msg = 'Lpix: {:.5f}'.format(median(lengthPixArray))
                cv.putText(frame, msg,
                           (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX,
                           1, RGB.Black, 2)

                msg = '{:4d}.00 mm'.format(round(lengthPix / (333.66 / 210)))
                cv.putText(frame, msg,
                           (int(topLx), int(topLy)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           1.5, RGB.Lime, 2)

    return frame


cv.namedWindow(WINDOWS_NAME)
cv.startWindowThread()

while True:
    if stream.die is True:
        break

    frame = stream.readClean()

    if frame is not None:
        # croop, to area of interest
        height, width = frame.shape[:2]
        frame = frame[600:1200, 1600:2500]
        height, width = frame.shape[:2]
        centerY, centerX = (int(height / 2), int(width / 2))

        frame = cv.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
        hsvF = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        hFrame, medianRange = scanColorRangeMedian(
            hsvF,
            centerX, centerY,
            matX=25, matY=25,
            draw=True
        )

        hMask = cv.inRange(hsvF, medianRange - 15, medianRange + 15)

        canny = cv.Canny(hMask, 50, 120)
        target = rectangleSize(hMask, frame)

        hFrame = cv.cvtColor(hFrame, cv.COLOR_HSV2BGR)
        hMask = cv.cvtColor(hMask, cv.COLOR_GRAY2BGR)
        canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

        frame = np.hstack((
            np.vstack((hFrame, hMask)),
            np.vstack((canny, target))
        ))

        if (SCALE_ARGS > 1):
            frame = downScaleImage(frame, SCALE_ARGS)

        frame = cv.putText(frame, 'FPS:{:3d}'.format(stream.readFPS()),
                           (20, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)

        cv.imshow(WINDOWS_NAME, frame)
        k = cv.waitKey(int(1000 / FPS_ARGS))

        if k == 27:
            break

stream.stop()
cv.destroyAllWindows()

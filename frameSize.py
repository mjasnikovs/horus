import cv2 as cv
import numpy as np
import argparse

from statistics import median
from helper import webcamStream, downScaleImage, scanColorRange, scanColorRangeMedian, scanColorDominant, RGB
from imutils import contours, perspective
from scipy.spatial import distance
from operator import itemgetter

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

# width
pixelsPerMetricX = 3.849631579
# length
pixelsPerMetricY = 3.913043478


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def processFrame(frame):
    # croop, to area of interest
    height, width = frame.shape[:2]
    frame = frame[frameCroopY[0]:frameCroopY[1], 0:width]
    height, width = frame.shape[:2]
    centerY, centerX = (int(height / 2), int(width / 2))

    # frame = cv.bilateralFilter(frame, 10, 60, cv.BORDER_WRAP)
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    tRange, lowRange, uppRange = scanColorRange(hsvFrame,
                                                centerX, centerY,
                                                matX=20, matY=3, draw=True)

    tMedian, medianRange = scanColorRangeMedian(hsvFrame,
                                                centerX, centerY,
                                                matX=100, matY=10,
                                                draw=True)

    tFrame, dominant = scanColorDominant(hsvFrame,
                                         centerX - 10, centerY - 20,
                                         200, 40,
                                         draw=True)

    fRange = cv.inRange(hsvFrame, lowRange - 25, uppRange + 25)
    fMedian = cv.inRange(hsvFrame, medianRange - 25, medianRange + 25)
    fFrame = cv.inRange(hsvFrame, dominant - 25, dominant + 25)

    fRange = cv.cvtColor(fRange, cv.COLOR_GRAY2BGR)
    tRange = cv.cvtColor(tRange, cv.COLOR_HSV2BGR)

    fMedian = cv.cvtColor(fMedian, cv.COLOR_GRAY2BGR)
    tMedian = cv.cvtColor(tMedian, cv.COLOR_HSV2BGR)

    fFrame = cv.cvtColor(fFrame, cv.COLOR_GRAY2BGR)
    tFrame = cv.cvtColor(tFrame, cv.COLOR_HSV2BGR)

    return np.vstack((fRange, tRange, fMedian, tMedian, fFrame, tFrame))


MATRIX = np.loadtxt('calibrations/' + NAME_ARGS + '_matrix.txt', delimiter=',')
DISTORTION = np.loadtxt('calibrations/' + NAME_ARGS + '_distortion.txt',
                        delimiter=',')

stream = webcamStream(1, WIDTH_ARGS, HEIGHT_ARGS, FPS_ARGS).start()
stream.calibrateCamera(MATRIX, DISTORTION)
stream.stream.set(cv.CAP_PROP_EXPOSURE, -5)

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

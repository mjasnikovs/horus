import cv2 as cv
import numpy as np
import argparse

from helper import webcamStream, downScaleImage, scanColorRangeMedian, RGB
from helper import midpoint

from imutils import perspective
from scipy.spatial import distance
from statistics import median

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv.boxPoints(rect)
    pts = np.int0(cv.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop


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
    stream.stream.set(cv.CAP_PROP_EXPOSURE, -3)

widthPixArray = list()
lengthPixArray = list()


def openClProcess(frame):
    frame = cv.UMat(frame)
    # frame = cv.bilateralFilter(frame, 10, 255, cv.BORDER_WRAP)
    frame = cv.bilateralFilter(frame, 25, 10, 255, borderType=cv.BORDER_WRAP)
    # frame = cv.Canny(frame, 65, 130)
    frame = cv.dilate(frame, None, iterations=2)
    frame = cv.erode(frame, None, iterations=2)
    return cv.UMat.get(frame)


def rectangleSize(mask, frame):
    global widthPixArray
    global lengthPixArray
    widthPix = 0
    lengthPix = 0

    frame = frame.copy()

    _, cnts, hier = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) < 1:
        return frame

    for c in cnts:
        area = cv.contourArea(c)

        if area < 10000:
            continue

        rect = cv.minAreaRect(c)
        angle = rect[2]
        box = cv.boxPoints(rect)
        tl, tr, br, bl = perspective.order_points(box)

        cv.circle(frame, tuple(tl), 5, RGB.Lime, -1)
        cv.circle(frame, tuple(tr), 5, RGB.Lime, -1)
        cv.circle(frame, tuple(br), 5, RGB.Lime, -1)
        cv.circle(frame, tuple(bl), 5, RGB.Lime, -1)

        # extract top and bottom coordinates of bounding area
        x, y, w, h = cv.boundingRect(c)

        # > Create canvas and rotate to proper angle
        # crop area of interest
        lenghtPiece = mask[int(y):int(y + h), int(x):int(x + w)]
        height, width = lenghtPiece.shape[:2]

        # create blank canvas image, with bigger size
        blankCanvas = np.zeros((height + 100, width + 100), np.uint8)

        # copy area of interest into blank canvas with offset
        blankCanvas[50:50 + height, 50:50 + width] = lenghtPiece
        ch, cw = blankCanvas.shape[:2]

        # rotate image to 90 degrees, if required
        if (angle > 0 or angle < 0):
            center = (cw / 2, ch / 2)
            if (abs(angle) > 45):
                angle += 90
            rot = cv.getRotationMatrix2D(center, angle, 1)
            blankCanvas = cv.warpAffine(blankCanvas, rot, (cw, ch))

        ch, cw = blankCanvas.shape[:2]

        # > Width measurment
        # extract 50px columns from middle
        crop = blankCanvas[0:ch, int(cw / 2) - 25:int(cw / 2) + 25]
        # count white pixels and get largest value
        widthPix = np.amax(np.sum(crop == 255, axis=0))
        widthPixArray.append(widthPix)

        # > Lenght mesurment
        # Crop by width measurment, 20 px of bottom
        topY = int((ch - widthPix) / 2)
        botY = int((ch - widthPix) / 2 + widthPix)
        crop = blankCanvas[topY:botY, 0:cw]
        cv.imshow('croop', crop)
        ch, cw = crop.shape[:2]
        cropTop = crop[5:10, 0:cw]
        cropBot = crop[ch - 10:ch, 0:cw]

        # cv.imshow('hello', np.vstack((cropTop, cropBot)))

        if (len(cropTop) and len(cropBot)):
            lengthPixTop = np.amax(np.sum(cropTop == 255, axis=1))
            lengthPixBot = np.amax(np.sum(cropBot == 255, axis=1))
        else:
            lengthPixTop = 0
            lengthPixBot = 0

        # count white pixels and get largest value
        lengthPix = (lengthPixTop + lengthPixBot) / 2
        lengthPixArray.append(lengthPix)

    width = widthPix / 1.686440678
    length = lengthPix / 1.659725566

    if (len(lengthPixArray) and len(widthPixArray)):
        print(median(widthPixArray), median(lengthPixArray))

    msg = '{:.1f} x {:.1f}'.format(width, length)
    cv.putText(frame, msg,
               (10, 80),
               cv.FONT_HERSHEY_SIMPLEX,
               3, RGB.Lime, 2)
    return frame


cv.namedWindow(WINDOWS_NAME)
cv.startWindowThread()

while True:
    if stream.die is True:
        break

    frame = stream.readClean()

    if frame is not None:
        # crop, to area of interest
        height, width = frame.shape[:2]
        frame = frame[880:1480, 50:width - 50]
        height, width = frame.shape[:2]
        centerY, centerX = (int(height / 2), int(width / 2))

        frame = openClProcess(frame)
        hsvF = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        hFrame, medianRange = scanColorRangeMedian(
            hsvF,
            centerX, centerY,
            matX=100, matY=2,
            draw=True
        )

        hMask = cv.inRange(hsvF, medianRange - 45, medianRange + 45)

        target = rectangleSize(hMask, frame)

        hFrame = cv.cvtColor(hFrame, cv.COLOR_HSV2BGR)
        hMask = cv.cvtColor(hMask, cv.COLOR_GRAY2BGR)

        frame = np.vstack((
            hFrame,
            hMask,
            target
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

import cv2 as cv
import argparse
import keyboard
import os
import time
import math
import numpy as np

from helper import downScaleImage, RGB

parser = argparse.ArgumentParser()

parser.add_argument('--downscale', type=int,
                    help='Downscale display image size')

parser.add_argument('--width', type=int, help='Camera resolution - width')
parser.add_argument('--height', type=int, help='Camera resolution - height')
parser.add_argument('--name', type=str, help='Camera name')

args = parser.parse_args()

SCALE_ARGS = 1 if (args.downscale is None) else args.downscale
WIDTH_ARGS = 4096 if (args.width is None) else args.width
HEIGHT_ARGS = 2160 if (args.height is None) else args.height
NAME_ARGS = 'camera' if (args.name is None) else args.name

WINDOWS_NAME = 'cameraCalibration'
TEMP_DIR = './TEMP'

NUM_ROWS = 9
NUM_COLS = 6
DIMENSION = 25  # mm

TOP = 'TOP'
TOP_LEFT = 'TOP_LEFT'
TOP_RIGHT = 'TOP_RIGHT'
RIGHT = 'RIGHT'
BOTTOM = 'BOTTOM'
BOTTOM_LEFT = 'BOTTOM_LEFT'
BOTTOM_RIGHT = 'BOTTOM_RIGHT'
LEFT = 'LEFT'
CENTER = 'CENTER'
CENTER_LEFT = 'CENTER_LEFT'
CENTER_RIGHT = 'CENTER_RIGHT'

FRAME_LIST = [
    TOP_LEFT,
    TOP,
    TOP_RIGHT,
    CENTER_LEFT,
    CENTER,
    CENTER_RIGHT,
    BOTTOM_LEFT,
    BOTTOM,
    BOTTOM_RIGHT
]

TARGET_LIST = [
    CENTER,
    TOP,
    RIGHT,
    BOTTOM,
    LEFT
]

# 'L', 'A', 'G', 'S'
FRAM_CODEC = cv.VideoWriter_fourcc('L', 'A', 'G', 'S')

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

files = os.listdir(TEMP_DIR)
for file in files:
    os.remove(TEMP_DIR + '/' + file)


def drawTargetPolygon(frame, w, h, direction=None, target=None):
    if target is TOP_LEFT:
        x = -w
        y = -h
    elif target is TOP:
        x = 0
        y = -h
    elif target is TOP_RIGHT:
        x = w
        y = -h
    elif target is CENTER_LEFT:
        x = -w
        y = 0
    elif target is CENTER:
        x = 0
        y = 0
    elif target is CENTER_RIGHT:
        x = w
        y = 0
    elif target is BOTTOM_LEFT:
        x = -w
        y = h
    elif target is BOTTOM:
        x = 0
        y = h
    elif target is BOTTOM_RIGHT:
        x = w
        y = h

    if direction is TOP:
        tl = [int(w * 1.1 + x), h + y]
        tr = [int(w * 2 * 0.95 + x), h + y]
        br = [w * 2 + x, h * 2 + y]
        bl = [w + x, h * 2 + y]
    elif direction is RIGHT:
        tl = [w + x, h + y]
        tr = [w * 2 + x, int(h * 1.1 + y)]
        br = [w * 2 + x, int(h * 2 * 0.95 + y)]
        bl = [w + x, h * 2 + y]
    elif direction is BOTTOM:
        tl = [w + x, h + y]
        tr = [w * 2 + x, h + y]
        br = [int(w * 2 * 0.95 + x), h * 2 + y]
        bl = [int(w * 1.1 + x), h * 2 + y]
    elif direction is LEFT:
        tl = [w + x, int(h * 1.1 + y)]
        tr = [w * 2 + x, h + y]
        br = [w * 2 + x, h * 2 + y]
        bl = [w + x, int(h * 2 * 0.95 + y)]
    else:
        tl = [w + x, h + y]
        tr = [w * 2 + x, h + y]
        br = [w * 2 + x, h * 2 + y]
        bl = [w + x, h * 2 + y]

    points = np.array([tl, tr, br, bl], np.int32)
    cv.polylines(frame, [points], True, RGB.Olive, 2)
    return frame


def captureImages():
    IMAGE_COUNTER = 0
    vid = cv.VideoCapture(cv.CAP_DSHOW + 1)

    vid.set(cv.CAP_PROP_BRIGHTNESS, 128)
    vid.set(cv.CAP_PROP_CONTRAST, 128)
    vid.set(cv.CAP_PROP_SATURATION, 128)
    vid.set(cv.CAP_PROP_SHARPNESS, 128)
    vid.set(cv.CAP_PROP_GAIN, 0)
    vid.set(cv.CAP_PROP_ZOOM, 0)
    vid.set(cv.CAP_PROP_EXPOSURE, -3)

    vid.set(cv.CAP_PROP_CONVERT_RGB, 1)
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)

    vid.set(cv.CAP_PROP_AUTOFOCUS, 0)
    vid.set(cv.CAP_PROP_FOCUS, 0)

    vid.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH_ARGS)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT_ARGS)

    # vid.set(cv2.CAP_PROP_SETTINGS, 1)
    vid.set(cv.CAP_PROP_FOURCC, FRAM_CODEC)
    vid.set(cv.CAP_PROP_FPS, 30)

    cv.namedWindow(WINDOWS_NAME)
    cv.startWindowThread()

    while(True):
        IMWRITE = False
        ret, frame = vid.read()

        if not ret:
            vid.release()
            print('Camera release')
            exit(1)

        if keyboard.is_pressed(' '):
            cv.imwrite(TEMP_DIR + '/picture' +
                       str(IMAGE_COUNTER) + '.jpg', frame)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            IMAGE_COUNTER += 1
            IMWRITE = True

        frame = downScaleImage(frame, SCALE_ARGS)
        frame = cv.flip(frame, 1)

        cv.putText(frame,
                   'Press SPACE to take a picture',
                   (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.50, RGB.Green, 1)

        height, width = frame.shape[:2]
        heightFac = int(height / 3)
        widthFac = int(width / 3)

        # draw horizontal lines
        cv.line(frame, (0, heightFac),
                (width, heightFac), RGB.Lime, 1)
        cv.line(frame, (0, int(heightFac * 2)),
                (width, int(heightFac * 2)), RGB.Lime, 1)

        # draw vertical lines
        cv.line(frame, (widthFac, 0),
                (widthFac, height), RGB.Lime, 1)
        cv.line(frame, (int(widthFac * 2), 0),
                (int(widthFac * 2), height), RGB.Lime, 1)

        frame = drawTargetPolygon(frame, widthFac, heightFac,
                                  TARGET_LIST[IMAGE_COUNTER % 5],
                                  FRAME_LIST[math.trunc(IMAGE_COUNTER / 5)])

        cv.imshow(WINDOWS_NAME, frame)

        k = cv.waitKey(1)

        if k == 27 or IMAGE_COUNTER >= 44:
            break

        if IMWRITE is True:
            time.sleep(0.5)

    cv.destroyAllWindows()


def calibrateImages():
    files = os.listdir(TEMP_DIR)

    if (len(files) < 44):
        print('Error: Not enough images')
        return

    cv.namedWindow(WINDOWS_NAME)
    cv.startWindowThread()

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                DIMENSION, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NUM_ROWS * NUM_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NUM_COLS, 0:NUM_ROWS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    badImage = files[1]

    for image in files:
        img = cv.imread(TEMP_DIR + '/' + image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray,
                                                (NUM_COLS, NUM_ROWS), None)

        if ret is True:
            print('Press ESC to skip image or ENTER to accept')
            corners2 = cv.cornerSubPix(gray,
                                       corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(img, (NUM_COLS, NUM_ROWS), corners2, ret)

            cv.imshow(WINDOWS_NAME, img)
            k = cv.waitKey(0) & 0xFF

            if k == 27:  # ESC Button
                print('Image skipped')
                badImage = image
                continue

            print('Image accepted')
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            badImage = image

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1],
                                                      None,
                                                      None)

    image = cv.imread(TEMP_DIR + '/' + badImage)
    h, w = image.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,
                                                     dist,
                                                     (w, h),
                                                     1,
                                                     (w, h))

    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None,
                                            newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cv.imwrite(TEMP_DIR + '/' + NAME_ARGS + '_cal_result.png', dst)

    filename = TEMP_DIR + '/' + NAME_ARGS + '_Matrix.txt'
    np.savetxt(filename, mtx, delimiter=',')
    filename = TEMP_DIR + '/' + NAME_ARGS + '_Distortion.txt'
    np.savetxt(filename, dist, delimiter=',')

    mean_error = 0
    for i in range(len(objpoints)):
        imgpo, _ = cv.projectPoints(objpoints[i],
                                    rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpo, cv.NORM_L2) / len(imgpo)
        mean_error += error

    print('total error: ', mean_error / len(objpoints))


captureImages()
calibrateImages()

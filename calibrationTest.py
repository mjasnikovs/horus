import cv2 as cv
import numpy as np
import argparse
import os
from helper import downScaleImage

WINDOWS_NAME = 'cameraCalibration'
CAL_DIR = './calibrations'
NUM_ROWS = 9
NUM_COLS = 6
DIMENSION = 25  # mm

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
CAMERA_NAME = 'camera' if (args.name is None) else args.name

DISTORTION_FILE = CAL_DIR + '/' + CAMERA_NAME + '_distortion.txt'
MATRIX_FILE = CAL_DIR + '/' + CAMERA_NAME + '_matrix.txt'

FRAM_CODEC = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

if (os.path.isfile(DISTORTION_FILE) is False):
    print('No ' + DISTORTION_FILE + ' file')
    exit()

if (os.path.isfile(MATRIX_FILE) is False):
    print('No ' + MATRIX_FILE + ' file')
    exit()

DISTORTION = np.loadtxt(DISTORTION_FILE, delimiter=',')
MATRIX = np.loadtxt(MATRIX_FILE, delimiter=',')


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def playVideo():
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
    vid.set(cv.CAP_PROP_FPS, 5)

    cv.namedWindow(WINDOWS_NAME)
    cv.startWindowThread()

    while(True):
        ret, frame = vid.read()

        if not ret:
            vid.release()
            print('Camera release')
            exit(1)

        h, w = frame.shape[:2]
        cam_matrix, roi = cv.getOptimalNewCameraMatrix(MATRIX,
                                                       DISTORTION,
                                                       (w, h),
                                                       1,
                                                       (w, h))

        mapx, mapy = cv.initUndistortRectifyMap(MATRIX,
                                                DISTORTION,
                                                None,
                                                cam_matrix,
                                                (w, h),
                                                5)

        frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)

        frame = downScaleImage(frame, SCALE_ARGS)

        cv.imshow(WINDOWS_NAME, frame)

        k = cv.waitKey(1)

        if k == 27:
            break

    cv.destroyAllWindows()


playVideo()

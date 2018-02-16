import cv2 as cv
import numpy as np
from helper import webcamStream, downScaleImage
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--downscale', type=int,
                    help='Downscale display image size')

parser.add_argument('--width', type=int, help='Camera resolution - width')
parser.add_argument('--height', type=int, help='Camera resolution - height')
parser.add_argument('--fps', type=int, help='Target camera FPS')
parser.add_argument('--opencl', type=int, help='Use opencl')

args = parser.parse_args()

WINDOWS_NAME = 'camera reader example'
SCALE_ARGS = 1 if (args.downscale is None) else args.downscale
WIDTH_ARGS = 4096 if (args.width is None) else args.width
HEIGHT_ARGS = 2160 if (args.height is None) else args.height
FPS_ARGS = 120 if (args.fps is None) else args.fps

MATRIX = np.loadtxt('calibrations/brio1_matrix.txt', delimiter=',')
DISTORTION = np.loadtxt('calibrations/brio1_distortion.txt', delimiter=',')

stream = webcamStream(1, WIDTH_ARGS, HEIGHT_ARGS, FPS_ARGS).start()
stream.calibrateCamera(MATRIX, DISTORTION)

cv.namedWindow(WINDOWS_NAME)
cv.startWindowThread()

while True:
    frame = stream.readClean()

    if frame is not None:
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

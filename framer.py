import cv2 as cv
import numpy as np
import argparse

from helper import webcamStream, downScaleImage, contourPixels, RGB, createMask
from imutils import perspective

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
    stream.stream.set(cv.CAP_PROP_EXPOSURE, -5)

widthPixArray = list()
lengthPixArray = list()
widthPix = 0
lengthPix = 0

cv.namedWindow(WINDOWS_NAME)
cv.startWindowThread()

while True:
    if stream.die is True:
        break

    frame = stream.readClean()

    if frame is not None:
        # crop, to area of interest
        height, width = frame.shape[:2]
        frame = frame[400:1300, 0:width]
        height, width = frame.shape[:2]
        center = (width / 2, height / 2)

        frame, mask = createMask(
            frame,
            x=int(width / 2 - 400),
            y=int(height - 100),
            w=800,
            h=50,
            tolerance=40,
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
            rotation_matrix = cv.getRotationMatrix2D(center, angle + 90, 1)
            mask = cv.warpAffine(mask, rotation_matrix, (width, height))

            # crop bounding rectangle
            x, y, w, h = cv.boundingRect(c)
            crop = mask[y:y + w, x: x + w]
            ch, cw = crop.shape[:2]

            # crop height area of interest
            heightCrop = crop[0:ch, int(cw * 0.4):int(cw * 0.6)]
            _, heightPix, targ = contourPixels(heightCrop)

            if heightPix is not None:
                top, _, bottom = targ[:3]

                # crop width area of interest
                if heightPix >= 40:
                    widthCrop = crop[
                        int(top[1] + heightPix - 40):int(bottom[1]),
                        0:cw
                    ]
                    widthPix, _, _ = contourPixels(widthCrop)
                else:
                    widthPix = None

                heightPix = heightPix if heightPix is not None else 0
                widthPix = widthPix if widthPix is not None else 0

                # draw msg
                msg = '{:.1f} x {:.1f}'.format(heightPix, widthPix)
                cv.putText(
                    frame,
                    msg,
                    (50, 120),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2, RGB.Red, 2
                )

                # draw cm
                msg = '{:.1f} mm x {:.1f} mm'.format(
                    heightPix / 1.705263158,
                    widthPix / 1.683196919
                )
                cv.putText(
                    frame,
                    msg,
                    (50, 220),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2, RGB.Red, 2
                )

        frame = np.vstack((
            frame,
            cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
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

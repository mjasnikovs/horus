import cv2 as cv
import numpy as np

from threading import Timer, Thread
from scipy.stats import itemfreq
from imutils import contours, perspective
from scipy.spatial import distance


class RGB_INIT():
    def __init__(self):
        self.Black = (0, 0, 0)
        self.White = (255, 255, 255)
        self.Red = (0, 0, 255)
        self.Lime = (0, 255, 0)
        self.Blue = (255, 0, 0)
        self.Yellow = (0, 255, 255)
        self.Cyan = (255, 255, 0)
        self.Magenta = (255, 0, 255)
        self.Silver = (192, 192, 192)
        self.Gray = (128, 128, 128)
        self.Maroon = (0, 0, 128)
        self.Olive = (0, 128, 128)
        self.Green = (0, 128, 0)
        self.Purple = (128, 0, 128)
        self.Teal = (128, 128, 0)
        self.Navy = (128, 0, 0)


RGB = RGB_INIT()


class setInterval():
    ''' Execute function in interval

        Attributes:
            t           Time in seconds
            hFunction   Executed function
    '''

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def stop(self):
        self.thread.cancel()


class functionThread():
    ''' Execute function in new thread

        Attributes:
            hFunction   Executed function
    '''

    def __init__(self, hFunction):
        self.hFunction = hFunction
        self.die = False
        self.thread = Thread(target=self.run)

    def run(self):
        while not self.die:
            self.hFunction()

    def start(self):
        self.thread.start()

    def stop(self):
        self.die = True


class webcamStream():
    def __init__(self, src=0, width=4096, height=2160, fps=30):
        self.stream = cv.VideoCapture(cv.CAP_DSHOW + src)

        FRAM_CODEC = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

        self.stream.set(cv.CAP_PROP_BRIGHTNESS, 128)
        self.stream.set(cv.CAP_PROP_CONTRAST, 0)
        self.stream.set(cv.CAP_PROP_SATURATION, 128)
        self.stream.set(cv.CAP_PROP_SHARPNESS, 0)
        self.stream.set(cv.CAP_PROP_GAIN, 0)
        self.stream.set(cv.CAP_PROP_ZOOM, 0)
        self.stream.set(cv.CAP_PROP_EXPOSURE, -5)

        self.stream.set(cv.CAP_PROP_CONVERT_RGB, 1)
        self.stream.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)

        self.stream.set(cv.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv.CAP_PROP_FOCUS, 0)

        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        # vid.set(cv2.CAP_PROP_SETTINGS, 1)
        self.stream.set(cv.CAP_PROP_FOURCC, FRAM_CODEC)
        self.stream.set(cv.CAP_PROP_FPS, fps)

        (self.grabbed, self.frame) = self.stream.read()
        self.die = False
        self.calibrate = False
        self.FPS_COUNTER = 0
        self.FPS_VALUE = 0
        self.fpsInterval = None

    def start(self):
        Thread(target=self.update, args=()).start()
        self.fpsInterval = setInterval(1, self.fpsCounter)
        self.fpsInterval.start()
        return self

    def update(self):
        while True:
            if self.die:
                return

            (self.grabbed, frame) = self.stream.read()
            if self.calibrate is True:
                self.frame = cv.remap(frame,
                                      self.mapx, self.mapy, cv.INTER_LINEAR)
            else:
                self.frame = frame
            self.FPS_COUNTER += 1

            if not self.grabbed:
                self.stop()

    def read(self):
        return self.frame

    def readClean(self):
        frame = self.frame
        self.frame = None
        return frame

    def readFPS(self):
        return self.FPS_VALUE

    def stop(self):
        self.die = True
        self.frame = None
        if self.fpsInterval:
            self.fpsInterval.stop()
        self.stream.release()

    def fpsCounter(self):
        self.FPS_VALUE = self.FPS_COUNTER
        self.FPS_COUNTER = 0

    def calibrateCamera(self, MATRIX=None, DISTORTION=None):
        if MATRIX is None or DISTORTION is None:
            raise ValueError('MATRIX and DISTORTION can\'t be None')

        h, w = self.frame.shape[:2]

        cam_matrix, roi = cv.getOptimalNewCameraMatrix(
            MATRIX,
            DISTORTION,
            (w, h),
            1,
            (w, h))

        (self.mapx, self.mapy) = cv.initUndistortRectifyMap(
            MATRIX,
            DISTORTION,
            None,
            cam_matrix,
            (w, h),
            5)

        self.calibrate = True


def downScaleImage(image, scale):
    ''' Scale down image size

        Attributes:
            image   Image
            scale   Downscale division
    '''

    height, width = image.shape[:2]
    return cv.resize(image, (int(width / scale), int(height / scale)))


def scanColorRange(image, x=0, y=0,
                   matX=2, matY=2, matD=10,
                   draw=False):
    ''' Return lower and upper color range from pixel matrix coordinates

        Attributes:
            image   Image
            x       axis for start position
            y       axis for start position
            matX    point column amount
            matY    point row amount
            matD    point distance between eachother
            draw    draw point to image frame
    '''
    posY = int(y - matY / 2 * matD)
    posX = int(x - matX / 2 * matD)
    rangeList = np.array([], dtype=np.uint8).reshape(0, 3)

    if draw is True:
        frame = image.copy()
    else:
        frame = None

    for Y in range(0, matY):
        for X in range(0, matX):
            locX = int(posX + X * matD)
            locY = int(posY + Y * matD)
            rangeList = np.vstack([rangeList, image[locY, locX]])
            if draw is True:
                cv.circle(frame,
                          (locX, locY),
                          int(matD / 3), RGB.Cyan, -1)

    rangeList = np.sort(rangeList, axis=0)

    lowRange = rangeList[0]
    uppRange = rangeList[len(rangeList) - 1]

    return frame, lowRange, uppRange


def scanColorRangeMedian(image, x=0, y=0,
                         matX=2, matY=2, matD=10,
                         draw=False):
    ''' Return median color range from pixel matrix coordinates

        Attributes:
            image   Image
            x       axis for start position
            y       axis for start position
            matX    point column amount
            matY    point row amount
            matD    point distance between eachother
            draw    draw point to image frame
    '''
    posY = int(y - matY / 2 * matD)
    posX = int(x - matX / 2 * matD)
    rangeList = np.array([], dtype=np.uint8).reshape(0, 3)

    if draw is True:
        frame = image.copy()
    else:
        frame = None

    for Y in range(0, matY):
        for X in range(0, matX):
            locY = int(posY + Y * matD)
            locX = int(posX + X * matD)
            rangeList = np.vstack([rangeList, image[locY, locX]])
            if draw is True:
                cv.circle(frame,
                          (locX, locY),
                          int(matD / 3), RGB.Cyan, -1)

    return frame, np.median(rangeList, axis=0)


def scanColorDominant(image, x=0, y=0, w=2, h=2, draw=True):
    ''' Return dominant color from pixel matrix coordinates

        Attributes:
            image   Image
            x       x axis for start position
            y       y axis for start position
            w       width
            h       height
            draw    draw rectangle to image frame
    '''
    frame = image[y:y + h, x:x + w]

    arr = np.float32(frame)
    pixels = arr.reshape((-1, 3))

    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, centroids = cv.kmeans(pixels, int(w + h / 2), None,
                                     crit, 10, cv.KMEANS_RANDOM_CENTERS)

    palette = np.uint8(centroids)
    color = palette[np.argmax(itemfreq(labels)[:, -1])]

    if draw is True:
        image = cv.rectangle(
            image.copy(),
            (x, y),
            (x + w, y + h),
            RGB.Teal,
            -1
        )
        return image, color

    return None, color


def createMask(image, x=None, y=None, w=None, h=None, tolerance=15, draw=None):
    ''' Return mask image from defined by area of intrest HSV value

        Attributes:
            image       Image
            x           x axis for start position
            y           y axis for start position
            w           width
            h           height
            tolerance   +/- HSV value tolerance
            draw        draw target rectangle to image mask
    '''
    mask = image.copy()
    height, width = mask.shape[:2]

    x = x if x is not None else int(width / 2)
    y = y if y is not None else int(height / 2)
    w = w if w is not None else int(width * .1)
    h = h if h is not None else int(height * .1)

    mask = cv.cvtColor(mask, cv.COLOR_BGR2HSV)
    mask = cv.bilateralFilter(mask, 10, 10, 255, borderType=cv.BORDER_WRAP)
    mask = mask[:, :, 2]

    # crop area of interest and median HSV value key
    crop = mask[y:y + w, x:x + w]
    medVal = int(np.median(crop))

    # color threshold
    _, mask = cv.threshold(
        mask,
        medVal - tolerance,
        medVal + tolerance,
        cv.THRESH_BINARY
    )

    # draw HSV capture area
    if draw is True:
        cv.rectangle(image, (x, y), (x + w, y + h), RGB.Lime, 2)

    return image, mask


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def contourPixels(frame):
    _, cnts, hier = cv.findContours(
        frame.copy(),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        return (None, None, None)

    rect = cv.minAreaRect(cnts[0])
    box = cv.boxPoints(rect)
    tl, tr, br, bl = perspective.order_points(box)

    top = midpoint(tl, tr)
    bottom = midpoint(bl, br)
    left = midpoint(tl, bl)
    right = midpoint(tr, br)

    height = distance.euclidean(top, bottom)
    width = distance.euclidean(left, right)

    return width, height, (top, right, bottom, left)


def drawFrameSize(mask, frame, minArea=100000, pixMetricX=1, pixMetricY=1):
    frame = frame.copy()

    _, cnts, _ = cv.findContours(
        mask.copy(),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if len(cnts):
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")

        for c in cnts:
            if cv.contourArea(c) > minArea:
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
                dimA = dA / pixMetricX
                dimB = dB / pixMetricY

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

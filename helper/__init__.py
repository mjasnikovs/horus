import cv2 as cv
from threading import Timer, Thread


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
        self.stream.set(cv.CAP_PROP_CONTRAST, 128)
        self.stream.set(cv.CAP_PROP_SATURATION, 128)
        self.stream.set(cv.CAP_PROP_SHARPNESS, 128)
        self.stream.set(cv.CAP_PROP_GAIN, 0)
        self.stream.set(cv.CAP_PROP_ZOOM, 0)
        self.stream.set(cv.CAP_PROP_EXPOSURE, -3)

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
                self.die()

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

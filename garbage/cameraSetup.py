import cv2
import numpy as np
from threading import Timer, Thread

# Global stats
FRAME_FPS = 0
FRAME_FPS_COUNTER = 0

FRAME_PROCESS_FPS = 0
FRAME_PROCESS_FPS_COUNTER = 0

FRAME_CAPTURE_FPS = 0
FRAME_CAPTURE_FPS_COUNTER = 0

FRAME_W = 0
FRAME_H = 0
MESSAGE_F = 'FPS:{:3d} CFPS:{:3d} PFPS:{:3d} | RES: {:d} x {:d}'

# Global threads
VIDEO_FRAME = None
VIDEO_FRAME_BUFFER = None
VIDEO_CAP = None
VIDEO_STATS_THREAD = None
VIDEO_PROCESS_THREAD = None
VIDEO_CAPTURE_THREAD = None

# CV_FOURCC('P', 'I', 'M', '1')    = MPEG-1 codec
# CV_FOURCC('M', 'J', 'P', 'G')    = motion-jpeg codec (does not work well)
# CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
# CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
# CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
# CV_FOURCC('U', '2', '6', '3') = H263 codec
# CV_FOURCC('I', '2', '6', '3') = H263I codec
# CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec

FRAM_CODEC = cv2.VideoWriter_fourcc(cv2.CV_FOURCC_PROMPT)


class perpetualTimer():
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

    def cancel(self):
        self.thread.cancel()


class executionThread():
    def __init__(self, hFunction):
        self.hFunction = hFunction
        self.die = False
        self.thread = Thread(target=self.run)

    def run(self):
        while not self.die:
            self.hFunction()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.die = True


def videoStatus():
    global FRAME_FPS, FRAME_PROCESS_FPS, FRAME_CAPTURE_FPS
    global FRAME_FPS_COUNTER, FRAME_PROCESS_FPS_COUNTER
    global FRAME_CAPTURE_FPS_COUNTER

    FRAME_FPS_COUNTER = FRAME_FPS
    FRAME_PROCESS_FPS_COUNTER = FRAME_PROCESS_FPS
    FRAME_CAPTURE_FPS_COUNTER = FRAME_CAPTURE_FPS

    FRAME_FPS = 0
    FRAME_PROCESS_FPS = 0
    FRAME_CAPTURE_FPS = 0


VIDEO_STATS_THREAD = perpetualTimer(1, videoStatus)


def drawStats(frame):
    global FRAME_FPS_COUNTER, FRAME_W, FRAME_H
    global FRAME_PROCESS_FPS_COUNTER, FRAME_CAPTURE_FPS_COUNTER

    message = MESSAGE_F.format(FRAME_FPS_COUNTER,
                               FRAME_CAPTURE_FPS_COUNTER,
                               FRAME_PROCESS_FPS_COUNTER,
                               int(FRAME_W),
                               int(FRAME_H))
    cv2.putText(frame, message, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
    return frame


def resizeFrame(frame, scale):
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width / scale), int(height / scale)))


def captureVideoFrame():
    global VIDEO_CAP, VIDEO_FRAME, VIDEO_FRAME_BUFFER, FRAME_FPS
    global FRAME_CAPTURE_FPS

    if VIDEO_CAP is not None:
        ret, frame = VIDEO_CAP.read()

        if not ret:
            VIDEO_CAP.release()
            print('Camera release')

        FRAME_CAPTURE_FPS += 1
        VIDEO_FRAME_BUFFER = frame


def processVideoFrame():
    global VIDEO_FRAME, VIDEO_FRAME_BUFFER, FRAME_PROCESS_FPS

    if VIDEO_FRAME_BUFFER is not None:
        frame = VIDEO_FRAME_BUFFER
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Computer vision code
        frame = cv2.bilateralFilter(frame, 10, 50, cv2.BORDER_WRAP)
        frame = cv2.Canny(frame, 65, 130)
        frame = cv2.dilate(frame, None, iterations=1)
        frame = cv2.erode(frame, None, iterations=1)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = np.vstack((VIDEO_FRAME_BUFFER, frame))
        frame = resizeFrame(frame, 6)
        frame = drawStats(frame)

        FRAME_PROCESS_FPS += 1
        VIDEO_FRAME = frame
        VIDEO_FRAME_BUFFER = None


# video loop
def playvideo():
    global FRAME_FPS, FRAME_W, FRAME_H, VIDEO_CAP

    vid = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

    vid.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    vid.set(cv2.CAP_PROP_CONTRAST, 128)
    vid.set(cv2.CAP_PROP_SATURATION, 128)
    vid.set(cv2.CAP_PROP_SHARPNESS, 128)
    vid.set(cv2.CAP_PROP_GAIN, 0)
    vid.set(cv2.CAP_PROP_ZOOM, 0)
    vid.set(cv2.CAP_PROP_EXPOSURE, -3)

    vid.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

    vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    vid.set(cv2.CAP_PROP_FOCUS, 0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    # vid.set(cv2.CAP_PROP_SETTINGS, 1)
    vid.set(cv2.CAP_PROP_FOURCC, FRAM_CODEC)
    vid.set(cv2.CAP_PROP_FPS, 10)

    FRAME_W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('camera')
    cv2.startWindowThread()
    VIDEO_CAP = vid

    VIDEO_STATS_THREAD.start()

    VIDEO_PROCESS_THREAD = executionThread(processVideoFrame)
    VIDEO_PROCESS_THREAD.start()

    VIDEO_CAPTURE_THREAD = executionThread(captureVideoFrame)
    VIDEO_CAPTURE_THREAD.start()

    while(True):
        FRAME_FPS += 1

        if VIDEO_FRAME is not None:
            cv2.imshow('camera', VIDEO_FRAME)

        k = cv2.waitKey(100)

        if k == 27:
            break

    VIDEO_CAPTURE_THREAD.cancel()
    VIDEO_PROCESS_THREAD.cancel()
    VIDEO_STATS_THREAD.cancel()
    cv2.destroyAllWindows()


# init
playvideo()

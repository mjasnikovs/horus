import cv2
from threading import Timer

# Status
FRAME_FPS = 0
FRAME_FPS_COUNTER = 0
FRAME_W = 0
FRAME_H = 0
MESSAGE_F = 'FPS:{:3d} | RES: {:d} x {:d}'

# CV_FOURCC('P', 'I', 'M', '1')    = MPEG-1 codec
# CV_FOURCC('M', 'J', 'P', 'G')    = motion-jpeg codec (does not work well)
# CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
# CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
# CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
# CV_FOURCC('U', '2', '6', '3') = H263 codec
# CV_FOURCC('I', '2', '6', '3') = H263I codec
# CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec

FRAM_CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


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


def videoStatus():
    global FRAME_FPS, FRAME_FPS_COUNTER
    FRAME_FPS_COUNTER = FRAME_FPS
    FRAME_FPS = 0


t = perpetualTimer(1, videoStatus)
t.start()


def drawStats(frame):
    global FRAME_FPS_COUNTER, FRAME_W, FRAME_H
    message = MESSAGE_F.format(FRAME_FPS_COUNTER, int(FRAME_W), int(FRAME_H))
    cv2.putText(frame, message, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
    return frame


# video loop
def playvideo():
    global FRAME_FPS, FRAME_W, FRAME_H

    vid = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

    vid.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    vid.set(cv2.CAP_PROP_CONTRAST, 128)
    vid.set(cv2.CAP_PROP_SATURATION, 128)
    vid.set(cv2.CAP_PROP_SHARPNESS, 128)
    vid.set(cv2.CAP_PROP_GAIN, 0)
    vid.set(cv2.CAP_PROP_ZOOM, 0)
    vid.set(cv2.CAP_PROP_EXPOSURE, -5)

    vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

    vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    vid.set(cv2.CAP_PROP_FOCUS, 10)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #  vid.set(cv2.CAP_PROP_SETTINGS, 1)
    vid.set(cv2.CAP_PROP_FOURCC, FRAM_CODEC)
    vid.set(cv2.CAP_PROP_FPS, 90)

    FRAME_W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('camera')
    cv2.startWindowThread()

    while(True):

        ret, frame = vid.read()

        if not ret:
            vid.release()
            print('camera release')
            break

        FRAME_FPS += 1

        frame = drawStats(frame)

        cv2.imshow('camera', frame)

        k = cv2.waitKey(1)

        if k == 27:
            break

    t.cancel()
    cv2.destroyAllWindows()


# init
playvideo()

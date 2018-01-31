import cv2
import re
import numpy as np
import time

import imutils
from imutils import perspective
from imutils import contours

from pyzbar.pyzbar import decode

# config
scale = 1
frameWidth = 1920
frameHeight = 1080
windowsName = 'Window Name'

#video loop
def playvideo():
    vid = cv2.VideoCapture(0)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

    cv2.namedWindow(windowsName)
    cv2.startWindowThread()

    while(True):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print('release')
            break

        frame = processFrame(frame)

        cv2.imshow(windowsName, frame)

        k = cv2.waitKey(1)

        if k == 27:
            break

    cv2.destroyAllWindows()

def barcodeSearch(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = cv2.erode(img, None, iterations=1)

    code = str(decode(img))
    m = re.search('\d{7}\w{2}', code)

    if m:
        barcode = m.group()
        cv2.putText(frame, format(str(barcode)),
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 0), 2)

    return frame

#frame process
def processFrame(frame):
    barcode = barcodeSearch(frame)
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width / scale), int(height / scale)))

# init
playvideo()

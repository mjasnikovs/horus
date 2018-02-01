import cv2
import re
import numpy as np
import time
import socket

import imutils
from imutils import perspective
from imutils import contours

from pyzbar.pyzbar import decode

# config
scale = 1
frameWidth = 2304
frameHeight = 1536
windowsName = 'Window Name'

def sendBarcode(barcode):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.0.153', 55055))
    s.sendall(barcode.encode('utf-8'))
    s.shutdown(1)
    s.close()
    return

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

        frame, barcode = processFrame(frame)

        cv2.imshow(windowsName, frame)

        k = cv2.waitKey(1)

        if k == 27:
            break

        if (barcode):
            sendBarcode(barcode)
            time.sleep(5)

    cv2.destroyAllWindows()

def barcodeSearch(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = cv2.erode(img, None, iterations=1)

    code = str(decode(img))
    m = re.search('\d{7}\w{2}', code)

    if m:
        barcode = m.group()
        return (frame, barcode)

    return (frame, 0)

#frame process
def processFrame(frame):
    frame, barcode = barcodeSearch(frame)
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width / scale), int(height / scale)))

    if (barcode):
        cv2.rectangle(frame, (0, 0), (width, height), (255, 255, 255), -1)
        cv2.putText(frame, format(str(barcode)),
                        (50, int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    3, (0, 255, 0), 4)
    return (frame, barcode)

# init
playvideo()

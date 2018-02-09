# IMPORT DEFINITIONS
import cv2
import re
import numpy as np
import socket
import time
import datetime
import _thread
import pygame

from time import sleep
from threading import Timer
from pyzbar.pyzbar import decode

# WINDOW DEFINITION
WINDOWS_NAME = 'Barcode Reader'

# RESOLUTION DEFITIONS
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# RENDERING AND SCANNING DEFITIONS
SCALE = 1
DETECTION_INTERVAL = 1

# GLOBAL VARIABLES
sendFlag = True
barcodeBuffer = 0
firstFrame = 0
lastFrame = 0
lastBarcode = None
interval = 0
threadCount = 0
# bleepSound = "scanner2.wav"

# PROCESS DEFINITIONS
def diff_img(img0, img1, img2):
    d1 = cv2.absdiff(img2, img1)
    d2 = cv2.absdiff(img1, img0)
    return cv2.bitwise_and(d1, d2)

def reset_send_flag():
    global sendFlag
    sendFlag = True

def send_barcode(barcode):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.0.12', 55055))
    s.sendall(barcode.encode('utf-8'))
    s.shutdown(1)
    s.close()
    return

def play_video():
    global sendFlag, firstFrame, lastFrame, lastBarcode, interval, threadCount, prevFrame, prevPrevFrame, totalPixels, bleep
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    pygame.mixer.init(frequency=44200*2)
    # pygame.mixer.music.load(bleepSound)

    print ("Attempting to set codec to", fourcc)

    cap = cv2.VideoCapture(1)

    #out = cv2.VideoWriter('videos/output-' + str(int(time.time())) + '.avi', fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 255)
    cap.set(cv2.CAP_PROP_SATURATION, 0)
    cap.set(cv2.CAP_PROP_GAIN, 255)
    print("Codec set to", cap.get(cv2.CAP_PROP_FOURCC))

    cv2.namedWindow(WINDOWS_NAME)
    cv2.startWindowThread()

    while(True):
        if firstFrame == 0:
            firstFrame = time.time()
        if lastFrame == 0:
            lastFrame = time.time()
            
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            print('release')
            break
        
        duration = (time.time() - lastFrame)
        lastFrame = time.time()
        if duration == 0:
            duration = 0.00001
            
        fps =  int(1 / duration)
        fps = str(fps)
        duration = int(duration * 1000)
        duration = str(duration)

        elapsed = int(time.time() - firstFrame)

        # MANIPULATE AUTO FOCUS DISTANCE EVERY SECOND
        if elapsed % 2:
            cap.set(cv2.CAP_PROP_FOCUS, 0)
        else:
            cap.set(cv2.CAP_PROP_FOCUS, 150)
        elapsed = str(elapsed)

        #_thread.start_new_thread(detect_barcode, (str(elapsed), frame))
        detect_barcode(str(elapsed), frame)

        cv2.putText(frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " : " + elapsed, (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, "FPS " + fps, (8, 24 * 2), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, "T " + duration, (8, 24 * 3), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        frame = draw_barcode(frame, lastBarcode)

        # SCREEN ROTATION
        #rows, cols = frame.shape[:2]
        #m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        #dst = cv2.warpAffine(frame, m, (cols, rows))
               
        cv2.imshow(WINDOWS_NAME, frame)

        k = cv2.waitKey(1)

        if k == 27:
            break

        #if (barcode and sendFlag):
        #    sendFlag = False
        #    print(barcode)
        #    #send_barcode(barcode)
        #    t = Timer(0, reset_send_flag)
        #    t.start()
        #out.write(frame)
        
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
    
def barcode_search(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    code = str(decode(img))
    m = re.search('\d*\d', code)
    if code != '[]':
        print (code)
        pygame.mixer.music.play()
        
    if m:
        barcode = m.group()
        return (frame, barcode)

    return (frame, 0)

def draw_barcode(frame, barcode):
    global SCALE
    height, width = frame.shape[:2]
    if SCALE != 1:
        frame = cv2.resize(frame, (int(width / SCALE), int(height / SCALE)))
    cv2.putText(frame, format(str(barcode)), (8, height - 16), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return frame

def detect_barcode(threadName, frame):
    global interval, lastBarcode, threadCount
    threadCount += 1
    
    interval = (interval + 1) % DETECTION_INTERVAL

    barcode = None

    if interval == 0:
        frame, barcode = process_frame(frame)
        if barcode != 0:
            lastBarcode = barcode
        else:
            lastBarcode = None
    threadCount -= 1

def process_frame(frame):
    global sendFlag
    global barcodeBuffer

    if (sendFlag):
        frame, barcode = barcode_search(frame)
        if (barcode):
            barcodeBuffer = barcode
        return (frame, barcode)
    else:
        return (frame, 0)

# INITIATE PRIMARY PROCESS
play_video()

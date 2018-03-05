import cv2 as cv
import csv
import re
import numpy as np
import argparse
import serial

from helper import webcamStream, downScaleImage, RGB
from os import listdir, path
from pyzbar.pyzbar import decode
from threading import Timer

CSV_DIR = './CSV'
CSV_LIST = dict()

parser = argparse.ArgumentParser()

parser.add_argument('--downscale', type=int,
                    help='Downscale display image size')

parser.add_argument('--width', type=int, help='Camera resolution - width')
parser.add_argument('--height', type=int, help='Camera resolution - height')
parser.add_argument('--fps', type=int, help='Target camera FPS')
parser.add_argument('--name', type=str, help='Camera name')

args = parser.parse_args()

WINDOWS_NAME = 'CNC BARCODE READER'
SCALE_ARGS = 1 if (args.downscale is None) else args.downscale
WIDTH_ARGS = 1280 if (args.width is None) else args.width
HEIGHT_ARGS = 720 if (args.height is None) else args.height
FPS_ARGS = 10 if (args.fps is None) else args.fps
NAME_ARGS = None if (args.name is None) else args.name

SEND_FLAG = False
ERR_MESSAGE = None
SUCCES_MESSAGE = None
PARAM_BUFFER = None

CSV_KEYS = [
    None,
    None,
    'XX',  # garums
    'YY',  # platums
    'ZZ',  # augstums
    None,
    None,
    None,
    None,
    'J',   # puse
    'OO',  # makros, eņgēs
    'L',   # X eņģes
    None,
    'N',   # X eņģes
    None,
    'P',   # X eņģes
    None,
    'R',   # X eņģes
    None,
    'T',   # X eņģes
    'OO',  # makros, pretplāksne
    'V',   # X pretplāksne
    None,
    'X',   # X pretplāksne,
    'Y',   # urbumu Z
    'Z',   # urbumu X no malas
    'AA',  # urbumu Y
    'AB',  # urbumu solis
    'AC',  # urbumu Z
    'AD',  # urbumu X no malas
    'AE',  # urbumu Y
    'AF',   # urbumu solis
    None,
    None,
    None,
    None
]

stream = webcamStream(0, WIDTH_ARGS, HEIGHT_ARGS, FPS_ARGS).start()

if NAME_ARGS is not None:
    MATRIX = np.loadtxt('calibrations/' + NAME_ARGS + '_matrix.txt',
                        delimiter=',')
    DISTORTION = np.loadtxt('calibrations/' + NAME_ARGS + '_distortion.txt',
                            delimiter=',')
    stream.calibrateCamera(MATRIX, DISTORTION)

Port = serial.Serial(
    port='COM7',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)


def loadCSV():
    global CSV_LIST
    for file in listdir(CSV_DIR):
        if path.splitext(file)[1] == '.csv':
            f = open(CSV_DIR + '/' + file, 'r')
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                id = row[0]
                CSV_LIST[id] = dict()
                for key, val in enumerate(row):
                    if CSV_KEYS[key] is not None and val is not '':
                        CSV_LIST[id][CSV_KEYS[key]] = val
            f.close()


def resetSendFlag():
    global SEND_FLAG
    global ERR_MESSAGE
    global SUCCES_MESSAGE
    SEND_FLAG = False
    ERR_MESSAGE = None
    SUCCES_MESSAGE = None


def wrightCSV(barcode):
    global Port
    global CSV_LIST
    global SEND_FLAG
    global ERR_MESSAGE
    global SUCCES_MESSAGE
    global PARAM_BUFFER

    SEND_FLAG = True
    t = Timer(5, resetSendFlag)
    t.start()

    if barcode not in CSV_LIST:
        ERR_MESSAGE = 'Barcode not found in CSV files'
        return

    if PARAM_BUFFER is None:
        for key in CSV_LIST[barcode]:
            PARAM_BUFFER = key + '9="' + CSV_LIST[barcode][key] + '" '
            SUCCES_MESSAGE = barcode
            return
    else:
        for key in CSV_LIST[barcode]:
            PARAM_BUFFER += key + '5="' + CSV_LIST[barcode][key] + '" '
            SUCCES_MESSAGE = barcode

    if Port.is_open is False:
        Port.open()

    message = 'JW10*' + PARAM_BUFFER + '*\r\n'.upper()
    PARAM_BUFFER = None

    Port.write(str.encode(message))
    print(message)

    if Port.is_open is True:
        Port.close()

    return


loadCSV()

cv.namedWindow(WINDOWS_NAME)
cv.startWindowThread()

while True:
    if stream.die is True:
        break

    frame = stream.readClean()

    if frame is not None:
        height, width = frame.shape[:2]
        if SEND_FLAG is True:
            frame = np.zeros([height, width, 3], dtype=np.uint8)
            frame.fill(255)

            if ERR_MESSAGE is not None:
                cv.putText(
                    frame, str(ERR_MESSAGE),
                    (50, int(height / 2)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2, RGB.Red, 4)
            elif SUCCES_MESSAGE is not None:
                cv.putText(
                    frame, str(SUCCES_MESSAGE),
                    (50, int(height / 2)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2, RGB.Lime, 4)
        else:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.erode(gray, None, iterations=2)

            code = str(decode(gray))
            m = re.search('\d{7}\w{2}', code)

            if m:
                barcode = m.group()
                print(barcode)
                wrightCSV(barcode)

        if (SCALE_ARGS > 1):
            frame = downScaleImage(frame, SCALE_ARGS)

        cv.imshow(WINDOWS_NAME, frame)
        k = cv.waitKey(int(1000 / FPS_ARGS))

        if k == 27:
            break

stream.stop()
cv.destroyAllWindows()

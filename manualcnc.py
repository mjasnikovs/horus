import csv
import serial
from time import sleep
from os import listdir, path

CSV_DIR = './CSV'
CSV_LIST = dict()

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


def wrightCSV(barcode):
    global Port
    global CSV_LIST
    global SEND_FLAG
    global ERR_MESSAGE
    global SUCCES_MESSAGE
    global PARAM_BUFFER

    if barcode not in CSV_LIST:
        ERR_MESSAGE = 'Barcode not found in CSV files'
        return

    if PARAM_BUFFER is None:
        PARAM_BUFFER = ''
        for key in CSV_LIST[barcode]:
            if key is 'OO' or key is 'J':
                PARAM_BUFFER += key + '9="' + CSV_LIST[barcode][key] + '" '
            else:
                PARAM_BUFFER += key + '9=' + CSV_LIST[barcode][key] + ' '
        SUCCES_MESSAGE = barcode
        return
    else:
        for key in CSV_LIST[barcode]:
            if key is 'OO' or key is 'J':
                PARAM_BUFFER += key + '5="' + CSV_LIST[barcode][key] + '" '
            else:
                PARAM_BUFFER += key + '5=' + CSV_LIST[barcode][key] + ' '
        SUCCES_MESSAGE = barcode

    if Port.is_open is False:
        Port.open()

    message = 'JW10* ' + PARAM_BUFFER.upper() + '*\r\n'
    PARAM_BUFFER = None

    Port.write(str.encode(message))
    sleep(.1)
    Port.flush()
    print('LEN:', len(message), message)

    if Port.is_open is True:
        Port.close()

    return


loadCSV()
wrightCSV('1001450LH')
wrightCSV('1001450LS')

print('ERR_MESSAGE', ERR_MESSAGE)
print('SUCCES_MESSAGE', SUCCES_MESSAGE)

import cv2

windowsName = 'Window Name'

def playvideo():
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print('release')
            break

        frame = processFrame(frame)

        cv2.namedWindow(windowsName)
        cv2.startWindowThread()
        cv2.imshow(windowsName, frame)
        
        k = cv2.waitKey(0)

        if k == 27:
            break

    cv2.destroyAllWindows()

def processFrame(frame):
    return frame

playvideo()

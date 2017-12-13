import cv2
import math
import time
import numpy as np

mmRatio = 0.2979406021
scale = 2

windowsName = 'Window Name'

def playvideo():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
        
        k = cv2.waitKey(1)

        if k == 27:
            break

    cv2.destroyAllWindows()

def processFrame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)

    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # detect aproximinated contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

        if len(approx) == 4:
            # calculate area
            area = cv2.contourArea(approx)

            print(area)
            time.sleep(1)

            if (area >= 10000):
                # calculate angles
                x1, y1 = approx[0][0]
                x2, y2 = approx[3][0]
                angle1 = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)

                x1, y1 = approx[3][0]
                x2, y2 = approx[0][0]
                angle2 = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)

                if ((angle1 >= 87 and angle1 <= 93) and (angle2 >= 87 and angle2 <= 93)):
                    # use [c] insted [approx] for precise detection line
                    # c = c.astype("float")
                    # c *= ratio
                    # c = c.astype("int")
                    #  cv2.drawContours(image, [c], 0, (0, 255, 0), 3)
                    # (x, y, w, h) = cv2.boundingRect(approx)
                    # ar = w / float(h)

                    # draw detected object
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

                    # draw detected data 
                    M = cv2.moments(c)
                    if (M["m00"] != 0):
                        cX = int((M["m10"] / M["m00"]) * ratio)
                        cY = int((M["m01"] / M["m00"]) * ratio)

                        # a square will have an aspect ratio that is approximately
                        # equal to one, otherwise, the shape is a rectangle
                        # shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
                        (x, y, w, h) = cv2.boundingRect(approx)
                        ar = w / float(h)

                        # calculate width and height
                        width = w * mmRatio
                        height = h * mmRatio

                        messurment = '%0.2fmm * %0.2fmm' % (width, height)

                        # draw text
                        cv2.putText(frame, messurment, (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)



    blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    combined = np.vstack((blurred, frame))

    height, width = combined.shape[:2]
    return cv2.resize(combined, (int(width/scale), int(height/scale)))

playvideo()

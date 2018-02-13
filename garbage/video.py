import cv2
import math
import time
import numpy as np

mmRatio = 0.1479406021
scale = 2

frameWidth = 2304
frameHeight = 1536
frameCroopY = [650,950]

windowsName = 'Window Name'

def playvideo():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

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
    frame = frame[frameCroopY[0]:frameCroopY[1], 0:frameWidth]
    liveFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # liveFrame = cv2.medianBlur(liveFrame, 1)

    # frame, kernel, x, y
    # liveFrame = cv2.GaussianBlur(liveFrame, (9, 9), 0)

	# frame, sigmaColor, sigmaSpace, borderType
    liveFrame = cv2.bilateralFilter(liveFrame, 10, 50, cv2.BORDER_WRAP)

    # _, liveFrame = cv2.threshold(liveFrame, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    liveFrame = cv2.Canny(liveFrame, 75, 150, 9)
    
    # cv2.goodFeaturesToTrack(img,maxCorners,qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector)
    corners = cv2.goodFeaturesToTrack(liveFrame, 2000, 0.01, 10)

    if corners is not None:
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.rectangle(liveFrame, (x - 1, y - 1), (x + 1, y + 1), (255, 255, 255), -100)
           # cv2.circle(liveFrame, (x, y), 3, 255, -1)

    _, cnts, _ = cv2.findContours(
        liveFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # detect aproximinated contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # cv2.drawContours(frame, [approx], 0, (255, 0, 0), 1)

        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(approx) == 4:
            # calculate area
            area = cv2.contourArea(approx)
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 1)

            if (area >= 1000):
                cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)
                difference = abs(round(cv2.norm(approx[0], approx[2]) - cv2.norm(approx[1], approx[3])))

                if (difference < 30):
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
                        cX = int((M["m10"] / M["m00"]))
                        cY = int((M["m01"] / M["m00"]))

                        # a square will have an aspect ratio that is approximately
                        # equal to one, otherwise, the shape is a rectangle
                        # shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
                        (x, y, w, h) = cv2.boundingRect(approx)
                        ar = w / float(h)

                        # calculate width and height
                        width = w * mmRatio
                        height = h * mmRatio

                        messurment = '%0.2fmm * %0.2fmm | %s' % (width, height, difference)

                        # draw text
                        cv2.putText(frame, messurment, (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    liveFrame = cv2.cvtColor(liveFrame, cv2.COLOR_GRAY2BGR)
    combined = np.vstack((liveFrame, frame))

    height, width = combined.shape[:2]
    return cv2.resize(combined, (int(width/scale), int(height/scale)))

playvideo()

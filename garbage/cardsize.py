import cv2
import math

mmRatio = 0.2979406021

# config
scale = 2

# open
image = cv2.imread('./assets/card1.jpg')

ratio = image.shape[0] / float(image.shape[0])

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    # detect aproximinated contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 4:
        # calculate area
        area = cv2.contourArea(approx)

        if (area >= 100000):
            # calculate angles
            x1, y1 = approx[0][0]
            x2, y2 = approx[3][0]
            angle1 = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)

            x1, y1 = approx[3][0]
            x2, y2 = approx[0][0]
            angle2 = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)

            # print(angle1, angle2)

            if ((angle1 >= 87 and angle1 <= 93) and (angle2 >= 87 and angle2 <= 93)):
                # use [c] insted [approx] for precise detection line
                # c = c.astype("float")
                # c *= ratio
                # c = c.astype("int")
                #  cv2.drawContours(image, [c], 0, (0, 255, 0), 3)
                # (x, y, w, h) = cv2.boundingRect(approx)
                # ar = w / float(h)

                # draw detected object
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)

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
                    cv2.putText(image, messurment, (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # print('========>')

# resize
height, width = image.shape[:2]
image = cv2.resize(image, (int(width/scale), int(height/scale)))

# display
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

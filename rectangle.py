import cv2
import imutils
import numpy as np

# simplify
image = cv2.imread('square.jpg')
ratio = image.shape[0] / float(image.shape[0])

reverse = 255 - image
gray = cv2.cvtColor(reverse, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
        print(approx)

# display
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

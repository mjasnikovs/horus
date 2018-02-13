#import

import cv2
import numpy as np
import imutils
import re

from imutils import perspective
from imutils import contours
from pyzbar.pyzbar import decode  # pip install pyzbar

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# connfig
scale = 1.2

# open image
openImage = cv2.imread('./assets/barcode.jpg')
barcodeImage = openImage.copy()
img = cv2.cvtColor(openImage, cv2.COLOR_RGB2GRAY)
# img = cv2.bitwise_not(img)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur image
gradient = cv2.dilate(gradient, None, iterations=2)
gradient = cv2.erode(gradient, None, iterations=2)
gradient = cv2.bitwise_not(gradient)

_, cnts, _ = cv2.findContours(
	gradient.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts):
	cnts, _ = contours.sort_contours(cnts, method="left-to-right")

	for c in cnts:
		if cv2.contourArea(c) > 1000:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.04 * peri, True)

			if len(approx) == 4:
				box = cv2.minAreaRect(c)
				box = cv2.cv.BoxPoints(
					box) if imutils.is_cv2() else cv2.boxPoints(box)
				box = np.array(box, dtype="int")

				# order the points in the contour such that they appear
				# in top-left, top-right, bottom-right, and bottom-left
				# order, then draw the outline of the rotated bounding
				# box
				box = perspective.order_points(box)

				x = int(box[0][0] - 15)
				y = int(box[0][1] - 15)
				w = int(box[1][0] - box[0][0] + 30)
				h = int(box[2][1] - box[1][1] + 30)

				code = str(decode(barcodeImage[y:y + h, x:x + w]))
				m = re.search('\d{7}\w{2}', code)

				if m:
					code = m.group()
					cv2.rectangle(openImage, (x, y), (x + w, y + h), (255, 0, 255))
					cv2.putText(openImage, format(str(code)),
										(x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
										0.65, (255, 255, 255), 1)

img = gradient

# combine & resize
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

combined = np.hstack((openImage, img))

height, width = combined.shape[:2]
outputImage = cv2.resize(combined, (int(width / scale), int(height / scale)))

# display
cv2.imshow('image', outputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

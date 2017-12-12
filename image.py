import cv2

# config
scale = 4

image = cv2.imread('img_bad.jpg', 0)

# transform
imgageTransform = cv2.medianBlur(image, 5)
imgageTransform = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# resize
height, width = imgageTransform.shape[:2]
imgageTransform = cv2.resize(imgageTransform, (int(width/scale), int(height/scale)))

# display
cv2.imshow('image', imgageTransform)
cv2.waitKey(0)
cv2.destroyAllWindows()

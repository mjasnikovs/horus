import cv2 as cv
from helper import webcamStream, RGB
import time
import os

WINDOWS_NAME = 'CAPTURE SCREEN'
FOCUS = 30
IMAGE_COUNTER = 0

for root_dir, cur_dir, files in os.walk('./pictures'):
    IMAGE_COUNTER += len(files)

camera = webcamStream(cv.CAP_V4L2, 1280, 720, 60).start()
camera.stream.set(cv.CAP_PROP_CONTRAST, 128)
camera.stream.set(cv.CAP_PROP_AUTO_WB, 0)
camera.stream.set(cv.CAP_PROP_TEMPERATURE, 2000)
camera.stream.set(cv.CAP_PROP_FOCUS, FOCUS)


while(True):
    frame = camera.readClean()

    if frame is None:
        time.sleep(0.01)
        continue

    if camera.die is True:
        break

    k = cv.waitKey(1)

    # control focus -/+
    if k == 61 or k == 45:
        FOCUS = FOCUS + 1 if k == 61 else FOCUS - 1
        camera.stream.set(cv.CAP_PROP_FOCUS, FOCUS)

    # save image
    if k == 32:
        path = './pictures/capture_' + str(IMAGE_COUNTER) + '.jpg'
        cv.imwrite(path, frame)
        IMAGE_COUNTER += 1
        cv.putText(frame, 'PATH: ' + path, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, RGB.Green, 2)

        frame = cv.copyMakeBorder(
            frame,
            top=10,
            bottom=10,
            left=10,
            right=10,
            borderType=cv.BORDER_CONSTANT,
            value=RGB.Magenta
        )
        cv.imshow(WINDOWS_NAME, frame)
        k = cv.waitKey(3000)

    if k == 27:
        break

    fps = camera.readFPS()
    cv.putText(frame, 'FPS: ' + str(fps), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, RGB.Green, 2)
    cv.putText(frame, 'Focus: ' + str(FOCUS), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, RGB.Green, 2)
    cv.putText(frame, 'Images: ' + str(IMAGE_COUNTER), (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, RGB.Green, 2)

    cv.imshow(WINDOWS_NAME, frame)
       
camera.stop()
cv.destroyAllWindows()
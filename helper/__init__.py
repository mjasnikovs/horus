import cv2 as cv
from threading import Timer, Thread


class setInterval():
    ''' Execute function in interval

        Attributes:
            t           Time in seconds
            hFunction   Executed function
    '''

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def stop(self):
        self.thread.cancel()


class functionThread():
    ''' Execute function in new thread

        Attributes:
            hFunction   Executed function
    '''

    def __init__(self, hFunction):
        self.hFunction = hFunction
        self.die = False
        self.thread = Thread(target=self.run)

    def run(self):
        while not self.die:
            self.hFunction()

    def start(self):
        self.thread.start()

    def stop(self):
        self.die = True


def downScaleImage(image, scale):
    ''' Scale down image size

        Attributes:
            image   Image
            scale   Downscale division
    '''

    height, width = image.shape[:2]
    return cv.resize(image, (int(width / scale), int(height / scale)))


class RGB_INIT():
    def __init__(self):
        self.Black = (0, 0, 0)
        self.White = (255, 255, 255)
        self.Red = (255, 0, 0)
        self.Lime = (0, 255, 0)
        self.Blue = (0, 0, 255)
        self.Yellow = (255, 255, 0)
        self.Cyan = (0, 255, 255)
        self.Magenta = (255, 0, 255)
        self.Silver = (192, 192, 192)
        self.Gray = (128, 128, 128)
        self.Maroon = (128, 0, 0)
        self.Olive = (128, 128, 0)
        self.Green = (0, 128, 0)
        self.Purple = (128, 0, 128)
        self.Teal = (0, 128, 128)
        self.Navy = (0, 0, 128)


RGB = RGB_INIT()

import cv2 as cv


class RGBtoGray:
    """
                A class to convert RGB to Grayscale.

                ...

                Attributes
                ----------
                gray_img : Array
                    Grayscale image

                Class Method
                -------
                def toGray(cls, image):
                    Convert image to Grayscale.
                """
    def __init__(self, image):
        self.gray_img = image

    @classmethod
    def to_gray(cls, image):
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return cls(img)

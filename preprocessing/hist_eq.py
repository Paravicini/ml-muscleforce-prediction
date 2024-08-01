import cv2 as cv


class Hist_Eq:
    """
                A class to apply Histogram Equalization. Should increase Contrast.

                ...

                Attributes
                ----------
                eq_img : Array
                    Image after applying Histogram Equalization

                Class Method
                -------
                def apply_hist_eq(cls, image):
                    Applies Histogram Equalization. For RGB histogram_Eq. would be applied to each channel.
                """
    def __init__(self, image):
        self.eq_img = image

    @classmethod
    def apply_hist_eq(cls, image):
        img = cv.equalizeHist(image)
        return cls(img)

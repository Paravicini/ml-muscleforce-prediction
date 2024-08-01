from preprocessing.hist_eq import Hist_Eq
from preprocessing.crop import Crop
from preprocessing.rgbtogray import RGBtoGray
import torchvision.transforms as transforms
import logging


class Preprocess:
    """
                A class to apply preprocessing to images.

                ...

                Attributes
                ----------
                processed_img : Array
                    Image that has been cropped, converted to grayscale and histogram equalization has been applied.

                Class Method
                -------
                def preprocess(cls, image):
                    Applies RGBtoGray, Crop and Hist_Eq on image
                """

    def __init__(self, processed_img):
        self.processed_img = processed_img

    @classmethod
    def preprocess_frame(cls, image, preprocess=None):
        """
        Applies RGBtoGray, Crop and Hist_Eq on image
        """
        if preprocess is None:
            logging.info("preprocess is None! ")
            return cls(image)

        img = RGBtoGray.to_gray(image).gray_img
        img = Crop.crop(img).cropped_img
        if preprocess['hist_eq']:
            img = Hist_Eq.apply_hist_eq(img).eq_img
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([preprocess['img_mean']], [preprocess['img_std']])(img)
        if preprocess['resize']:
            img = transforms.Resize((preprocess['img_size'], preprocess['img_size']), antialias=False)(img)

        return cls(img)


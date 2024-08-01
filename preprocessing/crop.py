class Crop:
    """
            A class to crop images.

            ...

            Attributes
            ----------
            cropped_img : Array
                Cropped image of desired size. Set in Method.

            Class Method
            -------
            def crop(cls, image):
                Crop image. Possible for RGB and Greyscale.
            """

    def __init__(self, image):
        self.cropped_img = image

    @classmethod
    def crop(cls, image):
        cropped = image[85:505, 265:685]  # default: [50:540, 230:720]
        return cls(cropped)

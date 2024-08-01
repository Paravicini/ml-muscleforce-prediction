import cv2 as cv
from preprocessing.preprocess import Preprocess


class Vid:
    """
                A class to load and store the Images of an avi video file.

                ...

                Attributes
                ----------
                images : List of Numpy Arrays
                    List of Numpy Arrays that
                fps : int
                    Frames per secon
                nr_frames : int
                    How many frames does the whole video have
                vid_length : int
                    Length of video in seconds

                Class Method
                -------
                get_path(cls, root='data', label='Trials', end_folder='aligned'):
                    Get Path up to specified end_folder, of each Trial.
                """

    def __init__(self, images, fps, nr_frames):

        self.images = images
        self.fps = fps
        self.nr_frames = nr_frames
        self.vid_length = nr_frames / fps

    @classmethod
    def load_vid_file(cls, path, processed: dict = None):
        """
               Gets the Frames of the avi video file and stores them in a string.
               Additional meta-data about the video are also collected. Like fps and how many frames the video has.

               If the argument 'processed' is set to False preprocessing of the frames (images) will not be done.
               CAUTION: this will increase memory usage heavily as images are not cropped and converted to
               Grayscale then!

               Parameters
               ----------
               path : str
                   Path to avi video file

               processed : dict
                     Dictionary that contains information about preprocessing. If None, preprocessing will not be done.
                     Default is None. Meaning no preprocessing will be done.


               Returns
               -------
               Images (frames) of video, fps and nr_frames
               """
        video = cv.VideoCapture(f'{path}')
        fps = video.get(cv.CAP_PROP_FPS)
        nr_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
        if processed is not None:
            processed_images = []
            while video.isOpened():
                ret, frame = video.read()
                if ret:
                    processed_image = Preprocess.preprocess_frame(frame, preprocess=processed).processed_img
                    processed_images.append(processed_image)
                else:
                    break
            video.release()
            cv.destroyAllWindows()
            return cls(processed_images, fps, nr_frames)
        else:
            # no preprocessing
            images = []
            while video.isOpened():
                ret, frame = video.read()
                if ret:
                    image = frame
                    images.append(image)
                else:
                    break
            video.release()
            cv.destroyAllWindows()
            return cls(images, fps, nr_frames)

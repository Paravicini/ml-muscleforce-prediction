from data_loaders.mat import Mat
from data_loaders.vid import Vid
import glob
import os
import logging


class Trial:
    """
                A class to build Trial Objects. They hold the synchronized mat and vid files of a trial.

                ...

                Attributes
                ----------
                mat : Mat
                    Instance of class Mat, accessible Attributes are: mat_df, time, emg and force
                vid : Vid
                    Instance of class Vid, accessible Attributes are: images, fps, nr_frames and vid_length

                Class Method
                -------
                from_files(cls, trial_path, is_upper_case=False, processed=True):
                    Load Data from files using class Mat and Vid
                """
    def __init__(self, mat: Mat, vid):
        self.mat = mat
        self.vid = vid

    @classmethod
    def from_files(cls, trial_path: str, is_upper_case: bool = False, preprocessing: dict = None):
        """

        :param trial_path: Path to Trial (1 Path only)
        :param is_upper_case: Set to True if avi extensions of files are in Uppercase. Some avi reader are Case sensitive.
        :param preprocessing: Specify the preprocessing steps to be applied to the images. SHOULD NOT BE NONE
        :return: mat and vid instances
        :rtype: Mat, Vid instances
        """
        mat_path = Trial.get_path(trial_path, 'mat')
        vid_path = Trial.get_path(trial_path, 'AVI')

        mat_file = Mat.load_mat_file(mat_path, processed=preprocessing)

        if is_upper_case:  # avi reader is case sensitive
            vid_path = vid_path[:-3] + 'avi'

        vid_file = Vid.load_vid_file(vid_path, processed=preprocessing)

        return cls(mat_file, vid_file)

    @staticmethod
    def get_path(trial_path: str, extension: str):
        extension_file = os.path.join(trial_path, f'*.{extension}')
        path = glob.glob(extension_file)
        if len(path) == 0:
            logging.error(f'No .{extension} file found for trial:' + trial_path)

        if len(path) > 1:
            logging.warning(f'more than one .{extension} file! First was returned')
        return path[0]

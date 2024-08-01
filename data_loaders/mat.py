import pandas as pd
import scipy.io as sio
import torchvision.transforms as transforms


class Mat:
    """
        A class to represent the data in a mat file.

        ...

        Attributes
        ----------
        mat_df : pd_df
            Pd_df containing columns time, emg and force
        time : np.array
            Sampling times of emg data
        emg : np.array
            Emg data in [uV]
        force: np.array
            Force measurements [N]. Out of sync, because of muscle activation, time delay (250 us).

        Class Method
        -------
        load_mat_file(path, struct_name='data'):
            Loads data from mat file and returns instance of Mat
        """

    def __init__(self, time, emg, force):
        # self.mat_df = pd.DataFrame({'time': time, 'emg': emg, 'force': force})
        self.time = time
        self.emg = abs(emg)
        self.force = force

    @classmethod
    def load_mat_file(cls, path, struct_name='data', processed=None):
        """

        :param path: Path to mat file
        :type path: str
        :param struct_name: Name of 'Struct' in mat file
        :type struct_name: str
        :return: Numpy arrays of time, emg and force column of mat file.
        :rtype: Numpy arrays (1D)
        """
        mat_file = sio.loadmat(path, squeeze_me=True)
        time = mat_file[f'{struct_name}']['time'].item()
        emg = mat_file[f'{struct_name}']['emg'].item()
        #emg = (emg - processed['emg_mean']) / processed['emg_std']  # mean:= 0.0042, std:= 0.1927 (from dataset)
        force = mat_file[f'{struct_name}']['force'].item()

        return cls(time, emg, force)

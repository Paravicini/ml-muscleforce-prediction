import pandas as pd
import os


class Directory:
    """
            A class to find and store the Paths in a Directory.

            ...

            Attributes
            ----------
            trial_path : pd_df of strings
                Pd_df containing one column, called {label}, that holds in each row the path to a Trial.
                At each Trial Path we find the synchronised mat and avi (video) file of the Trial x.

            Class Method
            -------
            get_path(cls, root='data', label='Trials', end_folder='aligned'):
                Get Path up to specified end_folder, of each Trial.
            """
    def __init__(self, trial_path):
        self.trial_path = trial_path

    @classmethod
    def get_path(cls, root='data', label='Trials', end_folder='aligned'):
        """

        :param root: root folder or directory to file storage
        :type root: str
        :param label: label the df's column where paths will be stored as rows
        :type label: str
        :param end_folder: Folder in which Trial mat and avi files are stored in. !!! All trials need to share the same end folder!!
        :type end_folder: str
        :return: Pd_df with column {label} where rows are Trials paths up to end_folder
        :rtype: Pd_df of strings
        """
        d = []
        for path, subdirs, files in os.walk(f'{root}'):
            for name in subdirs:
                temp = os.path.join(path, name)
                d.append(temp)
        directory = pd.DataFrame(d, columns=[f'{label}'])
        directory = directory[directory[f'{label}'].str.endswith(f'{end_folder}', -1*len(end_folder))]
        #only store paths that end with {end_folder}
        return cls(directory.iloc[:, 0])
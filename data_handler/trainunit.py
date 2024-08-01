from torch import from_numpy, unsqueeze, as_tensor
import numpy as np


class TrainUnit:
    """
    Class to store the data for one training unit.
    Emg and Force are now converted to torch tensors.
    frame: torch tensor of shape (1, 1, Frame_size, Frame_size) : default size 224
    emg: torch tensor of shape (1, 1, 83)
    force: torch tensor of shape (1, 1) --> Average force
    """
    def __init__(self, frame, emg, force):
        self.frame = frame.unsqueeze(0)  # might have to change to list of frames
        self.emg = as_tensor(emg).unsqueeze(0).unsqueeze(0)
        self.force = as_tensor(np.array([force.mean()])).unsqueeze(0)

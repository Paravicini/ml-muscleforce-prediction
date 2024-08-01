from data_handler.trainunit import TrainUnit
import logging
from torch import as_tensor


def map_emg_to_frame(trials, emg_fps: int = 1000):
    """
    Maps BOTH emg and force to frame, assumes that sampling rate of emg and force are the same and
    bigger than video

    :param trials: List of Trial Objects with synchronized mat and vid files
    :param emg_fps: Sampling rate of EMG data
    :return: Mapped EMG and Force data to frames depends on the sampling rate of EMG and Video
    :rtype: List of TrainUnit Objects
    """
    if not isinstance(trials, list):
        trials = [trials]

    train_units_list = []
    for trial in trials:
        us_fps = trial.vid.fps
        vid_length = trial.vid.vid_length
        emg_length = trial.mat.time[-1]
        time_diff = vid_length - emg_length
        if time_diff < 0:
            logging.error("Video length is shorter than EMG length")
        surplus_frames = int(time_diff * us_fps) + 1
        frames = trial.vid.images[0: int(trial.vid.nr_frames - surplus_frames)]
        # map emg to frames
        fps_ratio = emg_fps / us_fps  # fps = frames per second
        for i in range(0, len(frames)):
            emg = trial.mat.emg[int(i * fps_ratio):int((i + 1) * fps_ratio)][:int(fps_ratio)]
            force = trial.mat.force[int(i * fps_ratio):int((i + 1) * fps_ratio)][:int(fps_ratio)]
            train_unit = TrainUnit(frames[i], emg, force)
            train_units_list.append(train_unit)
    return train_units_list


def get_image_stripes(trial_units, nr_stripes=3, width=64, mode='stripes'):
    new_units_list = []
    if mode == 'stripes':
        if nr_stripes % 2 == 0:
            logging.ERROR('Number of stripes is even, please choose uneven number of stripes')


        center = int(trial_units[0].frame.shape[-1] / 2)
        start = center - width * (nr_stripes - 1) / 2
        for trial_unit in trial_units:
            frame = trial_unit.frame.squeeze().numpy()
            for i in range(0, nr_stripes):
                stripe_start = int(start + i * int(width / 2))
                frame_stripe = frame[:, stripe_start:stripe_start + width]
                new_unit = TrainUnit(as_tensor(frame_stripe).unsqueeze(0), trial_unit.emg.squeeze(),
                                     trial_unit.force.squeeze())
                new_units_list.append(new_unit)
            del trial_unit

    elif mode == 'center':
        for trial_unit in trial_units:
            frame = trial_unit.frame.squeeze().numpy()
            width = int(frame.shape[-1] * 0.125)
            mid_point = int(frame.shape[-1] / 2)
            frame = frame[:, mid_point - width:mid_point + width]
            new_unit = TrainUnit(as_tensor(frame).unsqueeze(0), trial_unit.emg.squeeze(),
                                 trial_unit.force.squeeze())
            new_units_list.append(new_unit)
            del trial_unit

    return new_units_list

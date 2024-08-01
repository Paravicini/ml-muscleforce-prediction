import random
import numpy as np


def fake_forces(trials: list, fake_percentage):
    for trial in trials:
        forces = trial.mat.force
        for i in range(0, len(forces)):
            if bool_based_on_probability(fake_percentage):  # if true, then fake force rest keeps original force
                random_idx = np.random.randint(len(forces))
                trial.mat.force[i] = forces[random_idx]
    return trials


def bool_based_on_probability(probability=0.5):
    return random.random() < probability


def settings_configurator(model_type, SETTINGS:dict):
    if model_type in['end2end', 'encoded_US', 'encoded_regression', 'encoded_end2end']:
        DATASET_EMG = True
        DATASET_FRAME = True
        ONLY_US = False
        EMG_FEATURES = 83
        if model_type == 'encoded_regression':# 12 or 83 --> for plotting purposes
            EMG_FEATURES = 12
            PLOT_LABEL = 'Encoded sEMG and US Features'
        elif model_type == 'end2end':
            PLOT_LABEL = 'End to End NN'
        elif model_type == 'encoded_US':
            PLOT_LABEL = 'Encoded US Features'
        elif model_type == 'encoded_end2end':
            EMG_FEATURES = 12
            PLOT_LABEL = 'Encoded End to End NN'
    elif model_type == 'emg_regression':
        DATASET_EMG = True
        DATASET_FRAME = False
        ONLY_US = False
        EMG_FEATURES = 0
        PLOT_LABEL = 'sEMG Regression'
    elif model_type == 'US_regression':
        SETTINGS['mode'] = 'encoded_US'
        DATASET_EMG = False
        DATASET_FRAME = True
        ONLY_US = True
        EMG_FEATURES = 0
        PLOT_LABEL = 'US Regression'

    SETTINGS['dataset_emg'] = DATASET_EMG
    SETTINGS['dataset_frame'] = DATASET_FRAME
    SETTINGS['only_us'] = ONLY_US
    SETTINGS['emg_features'] = EMG_FEATURES
    SETTINGS['plot_label'] = PLOT_LABEL




"""
Old code

if FAKE_TRIALS:  # Check if model overfits
    forces = []  # get forces to then randomly assign them to trials
    for trial in trial_units:  #TODO: Change such that forces are swaped inside the trial
        forces.append(trial.force.squeeze().numpy())


    def bool_based_on_probability(probability=1 - FAKE_PERCENTAGE):
        return random.random() < probability


    for unit in trial_units:
        if not bool_based_on_probability(): # if true, then fake force rest keeps original force
            random_idx = np.random.randint(len(forces))
            unit.force = as_tensor(forces[random_idx]).unsqueeze(0).unsqueeze(0)
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from data_handler.map_emg_to_frame import map_emg_to_frame


def analyse_emg_pred(regressor, dataset, test_x, test_y, trials, params):
    print(f'Analysing predictions of {regressor.__class__.__name__}...')
    importances = regressor.feature_importances_

    # Plot feature importance
    fig0, ax0 = plt.subplots(figsize=(15, 10))
    ax0.bar(range(len(importances)), importances)
    ax0.set_title('Feature importance')
    ax0.set_ylabel('Importance')
    ax0.set_xlabel('Feature')
    fig0.tight_layout()
    plt.show()

    predictions = []
    ground_truth = []
    for i in range(len(dataset)):
        true_force = dataset.iloc[i, -1]
        ground_truth.append(true_force)
        pred = regressor.predict(dataset.iloc[i, :-1].to_numpy().reshape(1, -1)).reshape(1)[0]
        predictions.append(pred)

    plot_trials(regressor, trials[0], mode='train')
    plot_trials(regressor, trials[1], mode='val')


def plot_trials(regressor, trials, mode='train'):
    loss = nn.MSELoss()
    if mode == 'train':
        rows = 5
        cols = 3
        figsize = (15, 15)
    elif mode == 'val':
        rows = 1
        cols = 2
        figsize = (15, 5)

    fig1, ax1 = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig1.tight_layout(pad=2.0)
    for i, trial in enumerate(trials):
        mapped = map_emg_to_frame(trial)
        true_force = []
        predicted_forces = []
        for unit in mapped:
            # Get features
            emg = unit.emg.cpu().detach().squeeze().numpy().reshape(1, -1)
            pred_force = regressor.predict(emg).reshape(1)[0]
            predicted_forces.append(pred_force)
            true_force.append(unit.force.squeeze().numpy().reshape(1)[0])

        MSE = loss(torch.tensor(true_force), torch.tensor(predicted_forces))
        ax1 = plt.subplot(rows, cols, i + 1)
        ax1.plot(true_force, color='darkblue', label='Ground truth')
        ax1.plot(predicted_forces, color='orange', label='Predictions RMSE: {:.2f}'.format(np.sqrt(MSE)))
        ax1.legend()
    plt.show()

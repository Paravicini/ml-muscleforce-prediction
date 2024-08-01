import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from data_handler.map_emg_to_frame import map_emg_to_frame
from models.helper_functions import write_to_txt, add_horizontal_line_to_max_force


def analyse_pred(regressor, trainset, trials: list, params: dict, img_encoder=None,
                 emg_encoder=None):  # test_x, test_y,

    corr = show_corr(trainset, params, plot=False)

    importance = plot_importances(regressor, params, plot=False)

    fig1, ax1 = plt.subplots(figsize=(15, 10))
    ax1.bar(range(len(importance)), importance, label='Importance')
    if params["settings"]['dataset_emg']:
        ax1.axvspan(len(importance) - params["settings"]['emg_features'], len(importance), alpha=0.5,
                    color='cornflowerblue',
                    label='EMG Features')
    ax11 = ax1.twinx()
    ax11.bar(range(len(importance)), abs(corr), alpha=0.2, color='orange', label='Correlation to force')
    ax1.set_title(f'Feature importance and Correlation scores')
    ax1.set_ylabel(f'Importance Score')
    ax1.set_ylim([0, max(importance) * 1.1])
    ax1.set_xlabel(f'Feature (total {len(importance)})')
    ax11.set_ylabel(f'Correlation Score')
    ax11.set_ylim([0, 1.1 * max(abs(corr))])
    fig1.tight_layout(pad=2.0)
    ax1.legend(loc='upper left')
    ax11.legend(loc='upper right')
    plt.show()

    plot_trial_predictions(regressor, trials[0], params, mode='train', img_encoder=img_encoder, emg_encoder=emg_encoder,
                           plot_emg=params["settings"]['plot_emg'])
    plot_trial_predictions(regressor, trials[1], params, mode='val', img_encoder=img_encoder, emg_encoder=emg_encoder,
                           plot_emg=params["settings"]['plot_emg'])


def plot_importances(regressor, params, plot=True):
    # Get feature importance's
    importance = regressor.feature_importances_
    count = 0
    all = 0
    for imp in importance:
        all += 1
        if imp > 0:
            count += 1

    # Plot feature importance
    if plot:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.bar(range(len(importance)), importance, label='Importance')
        if params["settings"]['dataset_emg']:
            ax.axvspan(len(importance) - params["settings"]['emg_features'], len(importance), alpha=0.5,
                       color='cornflowerblue',
                       label='EMG Features')
        ax.set_title(f'Feature importance ({params["settings"]["plot_label"]})')
        ax.set_ylabel(f'Importance')
        ax.set_xlabel(f'Feature (total {all})')
        fig.tight_layout(pad=2.0)
        ax.legend()
        plt.show()
    else:
        return importance


def show_corr(dataset, params, plot=True):
    df = pd.DataFrame(dataset)
    corr = df.corr().iloc[:-1, -1]
    # to numpy
    corr = corr.to_numpy()
    if plot:
        # Create a bar chart of the correlations
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(range(len(corr)), corr)

        ax.set_ylabel('Correlation Coefficient')
        ax.set_xlabel('Features')
        ax.set_title(f'Correlation Coefficient by Feature ({params["settings"]["plot_label"]})')
        plt.tight_layout(pad=2.0)
        plt.show()
    else:
        return corr


def plot_trial_predictions(regressor, trials, parameters, mode='train', img_encoder=None, emg_encoder=None,
                           plot_emg=True):
    """
    Plot predictions of a regressor on a trial
    :param regressor: regressor to use
    :param img_encoder: encoder to use
    :param trials: list of trials
    :return: None
    """

    settings = parameters['settings']
    total_MSE = 0
    if mode == 'train':
        nrows = 5
        ncols = 3
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        # plt.tight_layout(pad=2.0)
    elif mode == 'val':
        nrows = 1
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4))
        # plt.tight_layout(pad=2.0)

    fig.tight_layout(pad=2.0)
    if img_encoder is not None:
        img_encoder.eval()
    if emg_encoder is not None:
        emg_encoder.eval()

    with torch.no_grad():
        for i, trial in enumerate(trials):
            mapped_trial_units = map_emg_to_frame(trial)
            true_force = []
            predicted_forces = []
            mean_emg = []
            for unit in mapped_trial_units:
                # Get features
                if unit.force > 50:
                    if settings['dataset_frame']:
                        if parameters['preprocessing']['stripes']:
                            frame = unit.frame[:, :, :, 96:160].to(f'{parameters["settings"]["device"]}')
                        else:
                            frame = unit.frame.to(f'{parameters["settings"]["device"]}')
                        encoded_img = img_encoder(frame)
                        frame_ftrs = encoded_img.cpu().detach().view(1, -1).squeeze().numpy().reshape(1, -1)
                    if settings['dataset_emg']:
                        if emg_encoder is not None:
                            if plot_emg:
                                mean_emg.append(abs(unit.emg.cpu().detach().squeeze().numpy()).mean())
                            emg = emg_encoder(unit.emg.type(torch.float32).to(f'{parameters["settings"]["device"]}'))
                            emg = emg.cpu().detach().squeeze().numpy().reshape(1, -1)

                        else:
                            if plot_emg:
                                mean_emg.append(abs(unit.emg.cpu().detach().squeeze().numpy()).mean())
                            emg = unit.emg.cpu().detach().squeeze().numpy().reshape(1, -1)

                    if settings['dataset_frame'] and settings['dataset_emg']:
                        features = np.concatenate((frame_ftrs, emg), axis=1)
                    elif settings['dataset_frame'] and not settings['dataset_emg']:
                        features = frame_ftrs
                    elif not settings['dataset_frame'] and settings['dataset_emg']:
                        features = emg

                    # Predict force
                    pred_force = regressor.predict(features.reshape(1, -1)).reshape(1)[0]

                    # print(f'pred_force shape: {pred_force.shape}')
                    predicted_forces.append(pred_force)
                    true_force.append(unit.force.squeeze().numpy().reshape(1)[0])

            total_MSE += MSE(true_force, predicted_forces)
            write_to_txt(i, pred_force, true_force, f'{mode}')
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.plot(true_force, color='darkblue', label='ŷ')
            ax.plot(predicted_forces, color='orange',
                    label=f'Pred \n (RMSE: {np.sqrt(MSE(true_force, predicted_forces)):.2f})')
            max_force = max(true_force)

            ax.set_ylabel('Force (N)')
            ax.set_xlabel('Sample Points')
            ax.set_xlim([0, len(true_force)])
            # add horizontal line at max force

            add_horizontal_line_to_max_force(true_force, max_force, ax)

            if plot_emg:
                ax1 = ax.twinx()
                ax1.plot(mean_emg, color='red', label='Mean EMG', alpha=0.3)
                # ax.set_title(f'{mode} trials')
                ax1.legend(loc='center left', bbox_to_anchor=(1.2, 0.7))
            ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.9))
            plt.tight_layout(pad=2.0)

        if mode == 'train':
            print(f'RMSE over all {mode} sample points: {np.sqrt(total_MSE / len(trials)):.2f}')
            fig.suptitle(
                f'Train Trial Predictions | Overall RMSE: {np.sqrt(total_MSE / len(trials)):.2f}   ({settings["plot_label"]})')
            plt.tight_layout(pad=2.0)
        else:
            print(f'RMSE over all {mode} sample points: {np.sqrt(total_MSE / len(trials)):.2f}')
            fig.suptitle(
                f'Validation Trial Predictions | Overall RMSE: {np.sqrt(total_MSE / len(trials)):.2f}   ({settings["plot_label"]})')
            plt.tight_layout(pad=2.0)

    plt.show()



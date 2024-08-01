import numpy as np
import torch
from matplotlib import pyplot as plt
from data_handler.map_emg_to_frame import map_emg_to_frame
from sklearn.metrics import mean_squared_error as MSE


from models.helper_functions import custom_plot, remove_some_ylabels, add_horizontal_line_to_max_force


# TODO: Calculate loss per trial and plot it
def visualise_trials(model, device, trials: list, val=False, params=None):
    """
    This function is used to perform advanced analysis on the model.
    :param model: The model that is used for the analysis
    :param trials: A list of trials that are used for the analysis
    :return:
    """

    total_RMSE = 0
    if val:
        rows = 1
        cols = 2
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 4))
    else:
        rows = 5
        cols = 3
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))

    fig.tight_layout(pad=2.0)
    model.eval()
    for i, trial in enumerate(trials):
        mapped = map_emg_to_frame(trial)
        true_force = []
        predicted_forces = []
        for unit in mapped:
            if unit.force > 50:
                if params['preprocessing']['stripes']:
                    center = unit.frame.shape[3] // 2
                    move = params['preprocessing']['stripe_width'] // 2
                    frame = unit.frame[:, :, :, center - move:center + move].type(torch.float32).to(device)
                else:
                    frame = unit.frame.type(torch.float32).to(device)
                emg = unit.emg.type(torch.float32).to(device)
                true_force.append(unit.force.squeeze().cpu().detach().numpy())
                predictions = model(emg, frame).cpu().detach().squeeze().numpy()
                predicted_forces.append(predictions)


        trial_RMSE = MSE(np.array(true_force), np.array(predicted_forces))
        total_RMSE += trial_RMSE
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(true_force, color='darkblue', label='Ground truth')
        ax.plot(predicted_forces, color='orange',
                label=f'Pred \n (RMSE: {np.sqrt(MSE(true_force, predicted_forces)):.2f})')
        max_force = max(true_force)
        if val == True:
            # to make the y labels less congested
            if i == 0:
                remove_some_ylabels(ax, how_many=2)
            else:
                remove_some_ylabels(ax, how_many=1)
        else:
            remove_some_ylabels(ax, how_many=2)
        ax.set_ylabel('Force (N)')
        ax.set_xlabel('Sample Point')
        ax.set_xlim([0, len(true_force)])

        add_horizontal_line_to_max_force(true_force, max_force, ax)
        ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.9))
        plt.tight_layout()

    total_RMSE = np.sqrt(total_RMSE / len(trials))
    if val:
        fig.suptitle(f'Validation Trials | Overall RMSE: {total_RMSE:.2f} | ({params["settings"]["plot_label"]})',
                     fontsize=16)
        plt.tight_layout(pad=2.0)
    else:
        fig.suptitle(f'Test Trials | Overall RMSE: {total_RMSE:.2f} | ({params["settings"]["plot_label"]})',
                     fontsize=16)
        plt.tight_layout(pad=2.0)

    plt.show()


def analyse_predictions(test_set, val_set, parameters: dict):
    """
    This function is used to perform advanced analysis on the model.
    :param parameters: dict with all parameters and settings
    :type parameters: dict
    :param model: The model that is used for the analysis
    :param val_set: The validation set that is used for the analysis
    :param test_set: The test set that is used for the analysis
    :return:
    """
    model = parameters['model']
    device = parameters['settings']['device']
    visualise_trials(model, device, val_set, val=True, params=parameters)
    visualise_trials(model, device, test_set, params=parameters)




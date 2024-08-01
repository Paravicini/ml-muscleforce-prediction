import torch
import logging
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms


def load_model(model, path):
    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()


def save_model_bool(epoch, val_loss, lowest_loss, params, threshold):
    if epoch == 0:
        return True
    elif (val_loss < lowest_loss) and ((lowest_loss - val_loss) > threshold):  # (lowest_loss / 10):
        return True
    else:
        return False


def save_model(model, optimizer, epoch, checkpoint, loss):
    path = checkpoint['path'] + checkpoint['prefix'] + str(checkpoint['counter']) + '.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    checkpoint['counter'] += 1
    return checkpoint


def save_best_model(model, optimizer, epoch_rmse, val_rmse, epoch, best_epoch, checkpoint, lowest_loss, loss,
                    show=None):
    # show is a df that is used to show the loss distribution its columns are output and label and loss (output-label)
    combined_loss = epoch_rmse + val_rmse
    lowest_combined_loss = lowest_loss[0] + lowest_loss[1]
    if epoch == 0:
        lowest_loss = (epoch_rmse, val_rmse)
        best_epoch = epoch
        checkpoint = save_model(model, optimizer, epoch, checkpoint, loss)
        if show is not None:
            best_df = show
    elif lowest_combined_loss > combined_loss and (
            lowest_combined_loss - combined_loss) > lowest_combined_loss / 20 and val_rmse < lowest_loss[1]:
        lowest_loss = (epoch_rmse, val_rmse)
        best_epoch = epoch
        checkpoint = save_model(model, optimizer, epoch, checkpoint, loss)
        if show is not None:
            best_df = show
    else:
        logging.info('No New Best Model Found')
        if show is not None:
            best_df = show
            best_epoch = best_epoch
            return lowest_loss, best_epoch, best_df, checkpoint
        else:
            return lowest_loss, best_epoch, checkpoint

    # different output for new best model for wiht show and without
    if show is not None:
        return lowest_loss, best_epoch, best_df, checkpoint
    else:
        return lowest_loss, best_epoch, checkpoint


def show_epoch_loss(epoch, params, epoch_loss, val_loss, df=None):  # outputs=None, labels=None):
    if df is not None:
        df['loss'] = df['output'] - df['label']
        print('Epoch [{}/{}], Epoch RMSE: {:.4f} | Validation RMSE: {:.4f}'.format(epoch + 1,
                                                                                   params['settings']['nn_epochs'],
                                                                                   epoch_loss, val_loss))
        return df
    else:
        print(
            'Epoch [{}/{}], EpochLoss: {:.4f} | Validation Loss: {:.4f}'.format(epoch + 1, params['settings']['epochs'],
                                                                                epoch_loss,
                                                                                val_loss))


def save_configs_to_txt(parameters: dict, model):
    time_stamp = parameters['time_stamp']
    with open(f'configs/configs_{time_stamp}.txt', 'w') as f:
        f.write(f'params: {parameters["settings"]}\n')
        f.write(f'model: {model}\n')


def write_to_txt(trial, pred_force, true_force, mode='train'):
    RMSE = np.sqrt(np.mean((pred_force - true_force) ** 2))
    max_pred = np.max(pred_force)
    max_true = np.max(true_force)
    min_pred = np.min(pred_force)
    min_true = np.min(true_force)
    std_pred = np.std(pred_force)
    std_true = np.std(true_force)
    mean_pred = np.mean(pred_force)
    mean_true = np.mean(true_force)
    if mode == 'train':
        with open(f'configs/results.txt', 'a') as f:
            f.write(f'TEST trial: {trial}\n')
            f.write(f'RMSE: {RMSE}\n')
            f.write(f'max_true: {max_true}  | max_pred: {max_pred}\n')
            f.write(f'min_true: {min_true} | min_pred: {min_pred}\n')
            f.write(f'std_true: {std_true} | std_pred: {std_pred}\n')
            f.write(f'mean_true: {mean_true} | mean_pred: {mean_pred}\n')
            f.write(f'\n')
    elif mode == 'val':
        with open(f'configs/results.txt', 'a') as f:
            f.write(f'Validation trial: {trial}\n')
            f.write(f'RMSE: {RMSE}\n')
            f.write(f'max_true: {max_true}  | max_pred: {max_pred}\n')
            f.write(f'min_true: {min_true} | min_pred: {min_pred}\n')
            f.write(f'std_true: {std_true} | std_pred: {std_pred}\n')
            f.write(f'mean_true: {mean_true} | mean_pred: {mean_pred}\n')
            f.write(f'\n')


def remove_ticks_and_labels(fig, ax):
    ax.xaxis.set_tick_params(labelbottom=False, bottom=False, top=False, right=False, )
    ax.yaxis.set_tick_params(labelleft=False, left=False, right=False, top=False, )
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.tight_layout()


def custom_plot(fig, ax):
    # Adjust tick label properties
    ax.tick_params(axis='both', labelsize=12, colors='gray', width=1, length=4)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    x_ticks = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

    y_ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    # plt.setp(ax.get_yticklabels(), fontsize=14)

    # Add gridlines
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Set plot background
    ax.set_facecolor('#F0F8FF')

    # Customize legend
    ax.legend(loc='upper right', fontsize=12, fancybox=True, framealpha=0.8)

    # Set title
    # ax.set_title('My Plot', fontsize=20)

    # Adjust layout to prevent label cutoff in Jupyter notebook
    fig.tight_layout()


def remove_some_ylabels(ax, how_many=1):
    if how_many == 1:
        # Get the current y-tick positions
        yticks = ax.get_yticks()

        # Remove the last tick label and position
        yticks = yticks[:-1]

        # Set the modified tick positions
        ax.set_yticks(yticks)

        # Get the current y-tick labels
        ytick_labels = ax.get_yticklabels()

        # Remove the last tick label
        ytick_labels[-1] = None

        # Set the modified tick labels
        ax.set_yticklabels(ytick_labels)
    elif how_many == 2:
        # Get the current y-tick positions
        yticks = ax.get_yticks()

        # Remove the last tick label and position
        yticks = yticks[:-2]

        # Set the modified tick positions
        ax.set_yticks(yticks)

        # Get the current y-tick labels
        ytick_labels = ax.get_yticklabels()

        # Remove the last tick label
        ytick_labels[-1] = None
        ytick_labels[-2] = None

        # Set the modified tick labels
        ax.set_yticklabels(ytick_labels)

def add_horizontal_line_to_max_force(true_force, max_force, ax):
    x_lim = np.where(true_force == max_force)[0][0]
    ax.hlines(y=max_force, xmin=0, xmax=x_lim, color='black', linestyle='--', alpha=0.5)
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, max_force, "{:.0f}".format(max_force), color="black", transform=trans,
            ha="right", va="center")
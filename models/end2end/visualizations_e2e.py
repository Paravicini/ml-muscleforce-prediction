from matplotlib import pyplot as plt
from models.helper_functions import custom_plot


def visualize_e2e(best_df, epoch_loss_list, val_loss_list, lowest_loss, params):
    fig1, ax1 = plt.subplots()
    ax1.plot(epoch_loss_list, color='darkblue', linewidth=2, label='Training Loss')
    ax1.plot(val_loss_list, color='orange', linewidth=2, label='Validation Loss')
    ax1.axhline(y=min(epoch_loss_list), color='black', linestyle='--', linewidth=0.5)
    custom_plot(fig1, ax1)
    fig1.suptitle(f'RMSE per Epoch | optimizer: {params["settings"]["optimizer"]} | Criterion: {params["settings"]["criterion"]} | ', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.legend()
    plt.tight_layout(pad=2)
    plt.show()

    loss_df = best_df['loss']
    fig5, ax5 = plt.subplots()
    fig5.suptitle(f'RMSE Distribution | optimizer: {params["settings"]["optimizer"]} | Criterion: {params["settings"]["criterion"]} | ')
    ax5.set_xlabel('Difference in [N]')
    ax5.set_ylabel('Repetitions')
    loss_df.hist(bins=100, color='darkblue', ax=ax5)
    plt.show()


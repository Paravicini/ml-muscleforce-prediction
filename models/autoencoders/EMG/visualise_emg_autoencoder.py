import torch
from matplotlib import pyplot as plt
from data_handler.map_emg_to_frame import map_emg_to_frame
from models.helper_functions import remove_ticks_and_labels


def visualise(emg_hat, reconstructed):
    fig, ax = plt.subplots()
    emg_hat = emg_hat[0, :, :].squeeze().cpu().detach().numpy()
    reconstructed = reconstructed[0, :, :].squeeze().cpu().detach().numpy()
    ax.plot(emg_hat, label='EMG_hat', linewidth=5)
    ax.plot(reconstructed, label='Reconstructed EMG', linewidth=5)
    remove_ticks_and_labels(fig, ax)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_reconstructed_emg_trials(encoder, decoder, trials, params):
    device = params["device"]
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Reconstructed EMG')
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for i, trial in enumerate(trials):
            mapped = map_emg_to_frame(trial)
            emg_hats = []
            reconstructed = []
            for unit in mapped:
                if unit.force > 10:
                    emg_hat = unit.emg.squeeze(1).type(torch.float32).to(device)
                    print(f'emg_hat shape: {emg_hat.shape}')
                    emg_hats.append(emg_hat)
                    encoded_emg = encoder(emg_hat)
                    reconstructed_emg = decoder(encoded_emg)
                    reconstructed.append(reconstructed_emg.squeeze())

            ax[i].plot(emg_hats, label='EMG_hat', linewidth=3)
            ax[i].plot(reconstructed, label='Reconstructed EMG', linewidth=3)
            ax[i].legend()
    plt.show()

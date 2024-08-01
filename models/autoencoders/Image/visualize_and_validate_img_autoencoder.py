import torch
from matplotlib import pyplot as plt
from torch import nn
from models.helper_functions import remove_ticks_and_labels


def show_reconstructed_img(label, reconstructed, params, epoch):
    label.numpy()
    reconstructed.numpy()
    settings = params['settings']
    if settings['mode'] in[ 'encoded_regression', 'encoded_end2end', 'encoded_US']:
        label = label[0, :, :]
        reconstructed = reconstructed[0, :, :]
    fig2, (ax2) = plt.subplots(1, 2)
    fig2.suptitle(f'Original vs. Reconstructed Image after {epoch} epochs')
    ax2[0].set_xlabel('Original')
    ax2[0].imshow(label, cmap='gray')
    ax2[1].set_xlabel('Reconstructed')
    ax2[1].imshow(reconstructed, cmap='gray')
    remove_ticks_and_labels(fig2, ax2[0])
    remove_ticks_and_labels(fig2, ax2[1])
    plt.show()


def validate_autoencoder(encoder, decoder, val_loader, params, epoch):
    encoder.eval()
    decoder.eval()
    criterion = nn.MSELoss()
    validation_loss = 0
    with torch.no_grad():
        for batch_idx, (frame, emg, force) in enumerate(val_loader):
            frame = frame.type(torch.float32).squeeze(1).to(f'{params["settings"]["device"]}')
            encoded = encoder(frame)
            reconstructed = decoder(encoded)
            loss = criterion(reconstructed, frame)
            validation_loss += loss.item()
    if epoch % 10 == 0:
        show_reconstructed_img(frame.cpu().detach().squeeze(), reconstructed.cpu().detach().squeeze(), params, epoch)
    validation_loss = validation_loss / batch_idx
    return validation_loss




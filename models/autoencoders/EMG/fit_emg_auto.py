import torch
from matplotlib import pyplot as plt
from models.autoencoders.EMG.visualise_emg_autoencoder import visualise
from models.helper_functions import save_model


def validate_emg_autoencoder(params, val_loader, epoch):
    params['emg_encoder'].eval()
    params['emg_decoder'].eval()
    val_loss = 0.0
    for batch_idx, (frame, emg, force) in enumerate(val_loader):
        # Prepare emg for model input
        label_emg = emg.type(torch.float32).squeeze(1).to(params['settings']['device'])

        # Forward pass
        encoded_emg = params['emg_encoder'](label_emg)
        reconstructed_emg = params['emg_decoder'](encoded_emg)

        # Calculate loss
        loss = params['criterion'](reconstructed_emg, label_emg)

        val_loss += loss.item()

    val_loss = val_loss / batch_idx
    return val_loss


def train_emg_autoencoder(train_loader, val_loader, parameters: dict):
    print(f'EMG Autoencoder training started')
    settings = parameters['settings']
    checkpoint = parameters['emg_checkpoint']
    parameters['emg_encoder'].train()
    parameters['emg_decoder'].train()

    lowest_loss = 1000000
    best_epoch = 0
    epoch_loss_list = []
    val_loss_list = []

    for epoch in range(settings['epochs']):
        epoch_loss = 0.0
        for batch_idx, (frame, emg, force) in enumerate(train_loader):
            # Prepare emg for model input
            label_emg = emg.type(torch.float32).squeeze(1).to(settings['device'])

            # Forward pass
            encoded_emg = parameters['emg_encoder'](label_emg)
            reconstructed_emg = parameters['emg_decoder'](encoded_emg)

            # Calculate loss
            loss = parameters['criterion'](reconstructed_emg, label_emg)

            # Backward pass
            parameters['emg_optimizer'].zero_grad()
            loss.backward()
            parameters['emg_optimizer'].step()

            # Print statistics
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            visualise(label_emg, reconstructed_emg)
        epoch_loss = epoch_loss / batch_idx
        epoch_loss_list.append(epoch_loss)

        # Validate model
        val_loss = validate_emg_autoencoder(parameters, val_loader, None)
        val_loss_list.append(val_loss)

        # Print statistics
        print(f'Epoch: {epoch + 1}/{settings["epochs"]}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save model
        if val_loss < lowest_loss and epoch % 10 == 0:
            checkpoint = save_model(parameters['emg_encoder'], parameters['emg_optimizer'], epoch, checkpoint, loss)

    fig, ax = plt.subplots()
    fig.suptitle('Loss over Epochs')
    ax.plot(epoch_loss_list, label='Training Loss')
    ax.plot(val_loss_list, label='Validation Loss')
    plt.legend()
    plt.show()
    print('Finished Training EMG Autoencoder')
    emg_path = checkpoint['path'] + checkpoint['prefix'] + str(checkpoint['counter']-1) + '.pt'
    return emg_path

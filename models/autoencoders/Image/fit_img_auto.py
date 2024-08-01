import torch
from matplotlib import pyplot as plt
from models.autoencoders.Image.visualize_and_validate_img_autoencoder import validate_autoencoder
from models.helper_functions import show_epoch_loss, save_model_bool, save_model


def train_img_autoencoder(train_loader, val_loader, parameters=None):

    optimizer = parameters['img_optimizer']
    criterion = parameters['criterion']
    device = parameters['settings']['device']
    checkpoint = parameters['img_checkpoint']
    encoder = parameters['img_encoder'].train()
    decoder = parameters['img_decoder'].train()

    print(f'Start training of IMG Autoencoder')

    lowest_loss = 1000000
    best_epoch = 0
    epoch_loss_list = []
    val_loss_list = []

    for epoch in range(parameters['settings']['epochs']):
        epoch_loss = 0
        for batch_idx, (frame, emg, force) in enumerate(train_loader):
            # Reshaping the image to (-1, 784)
            image = frame.type(torch.float32).squeeze(1).to(device)

            # Output of Autoencoder
            encoded_img = encoder(image)
            reconstructed = decoder(encoded_img)
            if batch_idx == 0 and epoch == 0 and parameters['settings']['debug']:
                print(f'input shape: {image.shape}')
                print(f'encoded shape: {encoded_img.shape}')
                print(f'flatt encoded: {encoded_img.view(encoded_img.shape[0], -1).cpu().detach().numpy().shape}')
                print(f'output shape: {reconstructed.shape}')
                print(f'emg shape: {emg.squeeze().cpu().detach().numpy().shape}')
                print(f'force shape: {force.squeeze().cpu().detach().numpy().shape}')

            # Calculating the loss function
            loss = criterion(reconstructed, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            ### DEBUGGING ###

        epoch_loss = epoch_loss / batch_idx
        epoch_loss_list.append(epoch_loss)

        val_loss = validate_autoencoder(encoder, decoder, val_loader, parameters, epoch)
        val_loss_list.append(val_loss)

        show_epoch_loss(epoch, parameters, epoch_loss, val_loss)

        if save_model_bool(epoch, val_loss, lowest_loss, parameters, threshold=0.01):
            lowest_loss = val_loss
            best_epoch = epoch
            if parameters['settings']['debug']:
                print(f'Saving model on epoch {epoch + 1}')
            checkpoint = save_model(encoder, optimizer, epoch, checkpoint, loss)

    print(f'Best Val Score: {lowest_loss} on Epoch {best_epoch + 1}')
    fig, ax = plt.subplots()
    fig.suptitle('Loss over Epochs')
    ax.plot(epoch_loss_list, label='Training Loss')
    ax.plot(val_loss_list, label='Validation Loss')
    plt.legend()
    plt.show()
    # Load the best model based on the checkpoint with the lowest loss
    best_model_path = checkpoint['path'] + checkpoint['prefix'] + str(checkpoint['counter'] - 1) + '.pt'
    return best_model_path

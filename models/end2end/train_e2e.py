import logging
import torch
import pandas as pd
import numpy as np
from models.end2end.prediction_analysis import analyse_predictions
from models.end2end.validation_loss_e2e import get_validation_loss
from models.end2end.visualizations_e2e import visualize_e2e
from models.helper_functions import save_best_model, show_epoch_loss, save_configs_to_txt, load_model


def train_and_validate_end2end(train_set, val_set, train_trials, validation_trials, parameters):
    print(f'Training of {parameters["settings"]["mode"]} model started on {parameters["settings"]["device"]}')

    model = parameters['model']

    if parameters is None:
        logging.error('No parameters were passed to the train function')

    optimizer = parameters['optimizer']
    criterion = parameters['criterion']
    device = parameters['settings']['device']
    checkpoint = parameters['model_checkpoint']

    lowest_loss = (1000000, 1000000)  # train, val
    best_epoch = 0
    epoch_loss_rmse = []
    val_loss_list = []

    for epoch in range(parameters['settings']['nn_epochs']):
        # Load in the data in batches using the train_loader object
        model.train()
        epoch_loss = 0.0
        epoch_squared_diffs = 0.0
        epoch_total_samples = 0.0
        for batch_idx, (frame, emg, force) in enumerate(train_set):

            images = frame.type(torch.float32).squeeze(1).to(device)
            emg = emg.type(torch.float32).squeeze(1).to(device)

            label = force.type(torch.float32).squeeze(1).to(device)
            # print(f'Label shape: {label.shape}')

            output = model(emg.float(), images.float())
            # print(f'Output shape: {output.shape}')

            loss = criterion(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                df = pd.DataFrame(np.concatenate((output.cpu().detach().numpy(), label.cpu().detach().numpy()), axis=1),
                                  columns=['output', 'label'])

            df = pd.concat([df, pd.DataFrame(
                np.concatenate((output.cpu().detach().numpy(), label.cpu().detach().numpy()), axis=1),
                columns=['output', 'label'])], ignore_index=True, axis=0)

            epoch_loss += loss.item()
            epoch_squared_diffs += torch.sum((output - label) ** 2)
            epoch_total_samples += images.shape[0]

        # Calculate training RMSE
        epoch_rmse = torch.sqrt(epoch_squared_diffs / epoch_total_samples).cpu().detach().numpy()
        epoch_loss_rmse.append(epoch_rmse)

        # Get Validation Loss
        val_rmse = get_validation_loss(model, val_set, device)

        val_loss_list.append(val_rmse)

        loss_df = show_epoch_loss(epoch, parameters, epoch_rmse, val_rmse, df)
        # lowest loss is validation loss
        lowest_loss, best_epoch, best_df, checkpoint = save_best_model(model, optimizer, epoch_rmse, val_rmse, epoch,
                                                                       best_epoch,
                                                                       checkpoint,
                                                                       lowest_loss, loss, loss_df)
    print('\n')
    print(f'Best Epoch: [{best_epoch + 1}] -->  Test Score: {lowest_loss[0]} Val Score: {lowest_loss[1]}')

    visualize_e2e(best_df, epoch_loss_rmse, val_loss_list, lowest_loss, parameters)

    # Load the best model based on the checkpoint with the lowest loss
    best_model_path = checkpoint['path'] + checkpoint['prefix'] + str(checkpoint['counter'] - 1) + '.pt'
    save_configs_to_txt(parameters, model)

    load_model(parameters['model'], best_model_path)
    analyse_predictions(train_trials, validation_trials, parameters=parameters)

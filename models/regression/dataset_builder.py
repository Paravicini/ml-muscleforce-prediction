import numpy as np
import pandas as pd
import torch


def build_dataset(model, data_loader, params, only_emg=False):
    """Get X and Y for regression.

    Args:
        model (torch.nn.Module): Model to get features from.
        data_loader (torch.utils.data.DataLoader): Data loader for data.
        params (dict): Parameters for the model.

    Returns:
        X (pd.DataFrame): Features.
        Y (pd.DataFrame): Labels.
    """
    # Get features
    data_df = pd.DataFrame()
    model.eval()
    with torch.no_grad():
        for batch_idx, (frame, emg, force) in enumerate(data_loader):
            # Get features
            frame = frame.to(f'{params["device"]}').squeeze(1)
            encoded = model(frame)
            batch_size = encoded.shape[0]
            frame_ftrs = encoded.cpu().detach().view(batch_size, -1).numpy()
            emg = emg.squeeze().cpu().detach().numpy()
            force = force.squeeze().cpu().detach().numpy()
            if batch_idx == 0:
                if only_emg:
                    data_df = pd.DataFrame(np.concatenate(emg, force), axis=1)
                else:
                    data_df = pd.DataFrame(np.concatenate((frame_ftrs, emg, force), axis=1))
            else:
                if only_emg:
                    data_df = pd.concat([data_df, pd.DataFrame(np.concatenate(emg, force), axis=1)], axis=0)
                else:
                    data_df = pd.concat([data_df, pd.DataFrame(np.concatenate((frame_ftrs, emg, force), axis=1))],
                                        axis=0)
    print(f'Data_loader shape: {len(data_loader) * params["batch_size"] - (params["batch_size"] - batch_size)})')
    print(f'X shape: {data_df.shape}')
    return data_df


def get_dataset(train_units: list, parameters, img_encoder=None, emg_encoder=None, ds_emg=True, ds_frame=True):
    """
    Get X and Y for regression. Input modalities are first encoded (if required) and then concatenated.
    Such that they can be used for regression.

    Args:
        img_encoder (torch.nn.Module): Encoder to get features from.
        train_units (list): List of units to get features from.
        parameters (dict): Parameters for the model.

    Returns:
        X (pd.DataFrame): Features.
        Y (pd.DataFrame): Labels.
    """
    # Get features
    data_df = pd.DataFrame()
    if img_encoder is not None:
        img_encoder.eval()

    with torch.no_grad():
        for i, unit in enumerate(train_units):
            # Get features
            if ds_frame:
                frame = unit.frame.to(f'{parameters["settings"]["device"]}')
                encoded_img = img_encoder(frame)
                frame_ftrs = encoded_img.cpu().detach().view(1, -1).squeeze().numpy().reshape(1, -1)
            if ds_emg:

                if emg_encoder is not None:
                    emg_encoder.eval()
                    encoded_emg = emg_encoder(unit.emg.type(torch.float32).to(f'{parameters["settings"]["device"]}'))
                    emg = encoded_emg.cpu().detach().view(1, -1).squeeze().numpy().reshape(1, -1)
                else:
                    emg = unit.emg.cpu().detach().squeeze().numpy().reshape(1, -1)

            force = unit.force.cpu().detach().squeeze().numpy().reshape(1, -1)

            if i == 0:
                if ds_frame and ds_emg:
                    if parameters['settings']['debug']:
                        print(f'frame_ftrs: {frame_ftrs.shape}')
                        print(f'emg: {emg.shape}')
                    data_df = pd.DataFrame(np.concatenate((frame_ftrs, emg, force), axis=1))
                elif ds_frame and not ds_emg:
                    data_df = pd.DataFrame(np.concatenate((frame_ftrs, force), axis=1))
                elif not ds_frame and ds_emg:
                    data_df = pd.DataFrame(np.concatenate((emg, force), axis=1))

            else:
                if ds_frame and ds_emg:
                    data_df = pd.concat([data_df, pd.DataFrame(np.concatenate((frame_ftrs, emg, force), axis=1))],
                                        axis=0)
                elif ds_frame and not ds_emg:
                    data_df = pd.concat([data_df, pd.DataFrame(np.concatenate((frame_ftrs, force), axis=1))],
                                        axis=0)
                elif not ds_frame and ds_emg:
                    data_df = pd.concat([data_df, pd.DataFrame(np.concatenate((emg, force), axis=1))],
                                        axis=0)

    return data_df


def build_emg_dataset(units):
    data_df = pd.DataFrame()
    for i, unit in enumerate(units):
        # Get features
        emg = unit.emg.cpu().detach().squeeze().numpy().reshape(1, -1)
        force = unit.force.cpu().detach().squeeze().numpy().reshape(1, -1)
        if i == 0:
            data_df = pd.DataFrame(np.concatenate((emg, force), axis=1))
        else:
            data_df = pd.concat([data_df, pd.DataFrame(np.concatenate((emg, force), axis=1))], axis=0)

    return data_df

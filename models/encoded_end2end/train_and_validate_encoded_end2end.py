import torch
from models.autoencoders.EMG.fit_emg_auto import train_emg_autoencoder
from models.autoencoders.Image.fit_img_auto import train_img_autoencoder
from models.encoded_end2end.encoded_end2end_model import MCEFNN
from models.encoded_end2end.fit_encoded_end2end import fit_encoded_end2end
from models.end2end.prediction_analysis import analyse_predictions
from models.helper_functions import load_model, save_configs_to_txt


def train_and_validate_encoded_end2end(train_loader, val_loader, train_trials, validation_trails, parameters):
    """
    This Model requires to be trained (theoretically also untrained) encoders for both input modalities. Thus, first the
    autoencoders are trained and then the end2end model is initialized with the encoders. Then the NN is trained.
    If pretrained encoders are to be used, set the parameters['settings']['train'] to False and set the paths to the
    pretrained models in the parameters['img_encoder_path'] and parameters['emg_encoder_path'].

    :type train_loader: torch.utils.data.DataLoader
    :type val_loader: torch.utils.data.DataLoader
    :param train_trials: train trials are needed for visualizing the predictions
    :param validation_trails: also needed for visualizing the predictions
    :param parameters: Dict with all the Settings
    :type parameters: dict
    """
    device = parameters['settings']['device']
    img_path, emg_path = None, None
    if parameters['settings']['train']:
        img_path = train_img_autoencoder(train_loader, val_loader, parameters=parameters)
        emg_path = train_emg_autoencoder(train_loader, val_loader, parameters=parameters)

    if parameters['settings']['validate']:
        if not parameters['settings']['train']:
            img_path = parameters["settings"]['img_encoder_path']
            emg_path = parameters["settings"]['emg_encoder_path']
        load_model(parameters['img_encoder'], img_path)
        load_model(parameters['emg_encoder'], emg_path)


    model = MCEFNN(parameters['img_encoder'], parameters['emg_encoder']).float().to(device)
    parameters['model'] = model
    optimizer = torch.optim.Adam(model.out_layer.parameters(), lr=0.075)
    parameters['optimizer'] = optimizer
    best_model_path = fit_encoded_end2end(train_loader, val_loader, parameters=parameters)
    save_configs_to_txt(parameters, None)
    load_model(model, best_model_path)
    analyse_predictions( train_trials, validation_trails, parameters=parameters)

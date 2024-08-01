from models.autoencoders.EMG.fit_emg_auto import train_emg_autoencoder
from models.autoencoders.Image.fit_img_auto import train_img_autoencoder
from models.helper_functions import save_configs_to_txt, load_model
from models.regression.fit_function import fit_regression


def train_and_validate_encoded_regression(train_loader, val_loader, train_units, validation_units, trials, parameters):
    img_path, emg_path = None, None
    if parameters['settings']['train']:
        img_path = train_img_autoencoder(train_loader, val_loader, parameters=parameters)
        emg_path = train_emg_autoencoder(train_loader, val_loader, parameters=parameters)

    if parameters['settings']['validate']:
        if not parameters['settings']['train']:
            img_path = parameters["settings"]['img_encoder_path']
            emg_path = parameters["settings"]['emg_encoder_path']
        save_configs_to_txt(parameters, None)
        load_model(parameters['img_encoder'], img_path)
        load_model(parameters['emg_encoder'], emg_path)
        fit_regression(parameters['regressor'], parameters['img_encoder'], train_units, validation_units, trials,
                       parameters, emg_encoder=parameters['emg_encoder'])

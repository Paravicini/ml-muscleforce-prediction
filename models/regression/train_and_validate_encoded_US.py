from models.autoencoders.Image.fit_img_auto import train_img_autoencoder
from models.helper_functions import save_configs_to_txt, load_model
from models.regression.fit_function import fit_regression
from models.regression.only_US.fit_US import fit_US_reg


def train_and_validate_encoded_US(train_loader, val_loader, parameters, train_units, validation_units, trials):
    if parameters['settings']['train']:
        best_model_path = train_img_autoencoder(train_loader, val_loader, parameters=parameters)
        print(f'Best model path: {best_model_path}')
    else:
        best_model_path = parameters["settings"]['img_encoder_path']
        print(f'Best model path: {best_model_path}')

    if parameters['settings']['validate'] and not parameters['settings']['only_us']:
        save_configs_to_txt(parameters, parameters['img_encoder'])
        load_model(parameters['img_encoder'], best_model_path)
        fit_regression(parameters['regressor'], parameters['img_encoder'], train_units, validation_units, trials, parameters)

    if parameters['settings']['only_us']:
        if not parameters['settings']['train']: # if you wish to load pretrained encoder
            best_model_path = parameters.get('img_encoder_path')
        load_model(parameters['img_encoder'], best_model_path)
        fit_US_reg(parameters['regressor'], parameters['img_encoder'], train_units, trials, parameters)


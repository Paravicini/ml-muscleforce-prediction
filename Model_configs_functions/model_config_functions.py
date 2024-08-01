import logging
import torch
from xgboost import XGBRegressor
import torch.nn as nn
from models.autoencoders.EMG.emg_decoder import EMGDecoder
from models.autoencoders.EMG.emg_encoder import EMGEncoder
from models.autoencoders.Image.image_decoder import ImageDecoder
from models.autoencoders.Image.image_encoder import ImageEncoder
from models.end2end.ForcePredictionNetwork import ForcePredictionNN


def initialize_model(parameters):
    model_type = parameters['settings']['mode']
    device = parameters['settings']['device']
    if model_type == 'end2end':
        model = ForcePredictionNN().float().to(device)
        parameters['model'] = model
        return model, None, None, None, None

    elif model_type in ['encoded_US', 'encoded_regression', 'encoded_end2end', 'US_regression', 'emg_regression']:

        img_encoder = ImageEncoder(stripes=parameters['preprocessing']['stripes']).float().to(device)
        img_decoder = ImageDecoder(stripes=parameters['preprocessing']['stripes']).float().to(device)
        parameters['img_encoder'] = img_encoder
        parameters['img_decoder'] = img_decoder

        emg_encoder = emg_decoder = None
        if model_type in ['encoded_regression', 'encoded_end2end']:
            emg_encoder = EMGEncoder().float().to(device)
            emg_decoder = EMGDecoder().float().to(device)
            parameters['emg_encoder'] = emg_encoder
            parameters['emg_decoder'] = emg_decoder

        return None, img_encoder, img_decoder, emg_encoder, emg_decoder

    else:
        logging.error('No valid mode was chosen. Please choose from end2end/ encoded_US/ encoded_regression/ emg_regression / US_regression')
        return None, None, None, None, None


def setup_regressor(parameters):
    model_type = parameters['settings']['mode']
    regressor = None
    if model_type in ['encoded_US', 'encoded_regression', 'emg_regression', 'US_regression']:
        if model_type == 'US_regression':
            model_type = 'encoded_US'
        regressor_config = {
            'encoded_US': (0.1, 3, 100, 0.2, 10, 150, 1, 0, 0.8, 33),
            'encoded_regression': (0.075, 3, 100, 0.2, 20, 10, 1, 0, 0.2, 33),
            'emg_regression': (0.1, 4, 200, 1, 3, 4.5, 1, 0, 0.125, 33)
        }
        config = regressor_config.get(model_type)
        if config:
            regressor = XGBRegressor(learning_rate=config[0], max_depth=config[1], n_estimators=config[2],
                                     subsample=config[3], reg_lambda=config[4], reg_alpha=config[5],
                                     min_child_weight=config[6], gamma=config[7], colsample_bytree=config[8],
                                     random_state=config[9])
            parameters['regressor'] = regressor

    return regressor


def choose_criterion(parameters):
    criterion_type = parameters['settings']['criterion']
    criterion_mapping = {
        'MSE': nn.MSELoss(reduction='mean'),
        'L1': nn.L1Loss(reduction='mean'),
        'HUBER': nn.HuberLoss(reduction='mean')
    }
    criterion = criterion_mapping.get(criterion_type)
    if criterion is None:
        logging.error('No valid criterion was chosen. Please choose from MSE/ L1/ HUBER')
    parameters['criterion'] = criterion
    return criterion


# Define a function to create an optimizer
def create_optimizer(params, optimizer_type, learning_rate, momentum=0):
    if optimizer_type == 'Adam':
        return torch.optim.Adam(params, lr=learning_rate)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
    else:
        logging.error('No valid optimizer was chosen. Please choose from Adam/ SGD')
        return None


def initialize_optimizer(parameters):
    mode = parameters['settings']['mode']
    optimizer_type = parameters['settings'].get('optimizer')

    if mode in ['encoded_US', 'encoded_regression', 'encoded_end2end']:
        img_encoder = parameters.get('img_encoder')
        img_decoder = parameters.get('img_decoder')
        if img_encoder and img_decoder:
            img_params = [{'params': img_encoder.parameters()}, {'params': img_decoder.parameters()}]
            parameters['img_optimizer'] = create_optimizer(img_params, optimizer_type, parameters['settings']['lr'], parameters['settings']['momentum'])

    if mode in ['encoded_regression', 'encoded_end2end']:
        emg_encoder = parameters.get('emg_encoder')
        emg_decoder = parameters.get('emg_decoder')
        if emg_encoder and emg_decoder:
            emg_params = [{'params': emg_encoder.parameters()}, {'params': emg_decoder.parameters()}]
            parameters['emg_optimizer'] = create_optimizer(emg_params, optimizer_type, parameters['settings']['lr'], parameters['settings']['momentum'])

    if mode == 'end2end':
        model = parameters.get('model')
        if model:
            model_params = model.parameters()
            parameters['optimizer'] = create_optimizer(model_params, optimizer_type, parameters['settings']['lr'], parameters['settings']['momentum'])

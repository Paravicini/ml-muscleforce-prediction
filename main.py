from datetime import datetime
from data_handler.balance_dataset import even_dataset
from data_handler.split_trials import split_trials
from data_loaders.directory import Directory
from data_handler.trial import Trial
import data_handler.map_emg_to_frame as func
from torch.utils.data import DataLoader
from data_handler.data_set import MyDataset
from models.encoded_end2end.train_and_validate_encoded_end2end import train_and_validate_encoded_end2end
from models.end2end.train_e2e import train_and_validate_end2end
import logging
from models.regression.only_emg.fit_emg import train_and_validate_emg_regression
from Model_configs_functions.model_config_functions import initialize_model, setup_regressor, choose_criterion, \
    initialize_optimizer
from models.regression.train_and_validate_encoded_US import train_and_validate_encoded_US
from models.regression.train_and_validate_encoded_regression import train_and_validate_encoded_regression
from toolbox.toolbox import settings_configurator, fake_forces

DEBUG = False

TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

logging.basicConfig(level=logging.INFO)

MEAN_FORCE = 540.9501  # Determined experimentally default 540.9501
MAX_FORCE = 2112
############################################################################################################
#           SET PARAMETERS FOR PREPROCESSING           #
############################################################################################################

# IMAGE PREPROCESSING
"""
Parameters for  img preprocessing: Image mean and std were calculated over all frames-pixels of the dataset. Value 
after Hist eq: 0.4999 (mean) and 0.2815 (std) before Hist eq: 0.1756 (mean) and 0.1135 (std). If dataset changes use 
get_data_statistics.ipynb to calculate new values.
Changing the img_size will change the size of the input image to the model. --> Change model architecture accordingly!

resize: True/False 
img_size: int # default 256
hist_eq: True/False 
img_mean: float 
img_std: float

Parameters for EMG preprocessing:
mean and std were calculated over all EMG values of the dataset 0.000011139 (mean) and 0.1295 (std).
    emg_mean: float
    emg_std: float
"""
RESIZE = True
IMG_SIZE = 256
HIST_EQ = True
IMG_MEAN = 0.4999
IMG_STD = 0.2851

# EMG PREPROCESSING
EMG_MEAN = 0.000011139
EMG_STD = 0.1295

# STRIPES
"""
In this section you can select if you wish to use stripes of your images instead of the full image. Choose between 
center and stripes mode. Center will crop each image to the center of the image. Stripes will create Nr_STRIPES 
stripes of width STRIPE_WIDTH. For each Frame, each stripe of the same Frame will share the same EMG and Force values.

Parameters for Stripes:
    stripes: True/False
    stripes_mode: 'center' / 'stripes'
    stripe_width: int
"""
STRIPES = True
STRIPES_MODE = 'center'  # choose from 'center' / 'stripes'
STRIPE_WIDTH = 64
Nr_STRIPES = 3

"""
This functionality can be used to balance the dataset. Very simplistic algorithm at the moment. We simply multiply 
under represented datapoints by a factor. Can/should be expanded in future...
"""
EVEN_DATASET = True

TRAIN_VAL_SPLIT = 0.9

"""
Using Fake Trials to check if model overfits. Fake Trials are created by randomly switching the force values between 
train_unit's for a given percentage (FAKE_Percentage) of all the train_units.
"""
FAKE_TRIALS = False
FAKE_PERCENTAGE = 0.5
if FAKE_TRIALS:
    logging.info(f'Fake Trials is set to True!')

############################################################################################################
#           SET PARAMETERS FOR TRAINING           #
############################################################################################################
"""
As the initial goal of this project was to investigate possible architectures for the muscle prediction task, 
the user can choose between different models. The following models are available:
    - end2end:              A simple Mulitmodal-CNN that takes train_units (X:= Frame + EMG , Y:= Force) 
                            as input and predicts the force in a end to end neural network approach. 
    - encoded_US:           In this model only the frames per train_units are encoded via a trained autoencoder. 
                            The EMG is than concatenated to the frame latent space and this acts as input to 
                            the xgboost regressor. 
    - US_regression         A regressor that takes only the frames as input and predicts the force from this. 
                            Uses an autoencoder to generate a latent space for input frames.
    - encoded_regression:   Both frames and EMG per train_unit are encoded via a trained autoencoders. 
                            Both latent representations are than concatenated and used as input to the xgboost regressor 
    - emg_regression:       A regressor that takes only the EMG signal as input and predicts the force from this
                            This is also a Reference Model configuration
    - encoded_end2end:      This model first trains or loads previously trained encoders into a end to end NN architecture. 
                            Within the NN the latent representations are concatenated and connected to a 
                            linear layer that predicts the force.
                            
   
"""

MODEL_TYPE = 'encoded_end2end'  # choose from 'end2end' / 'encoded_US' / 'US_regression' / 'encoded_regression' / 'emg_regression' / 'encoded_end2end'

# If Previously trained encoders should be used, set TRAIN_AUTOENCODER to False and VALIDATE to True
# and set the paths to the respective encoders!
TRAIN_AUTOENCODER = True
VALIDATE_AUTOENCODER = True
emg_encoder_path = 'models/autoencoders/EMG/trained_emg_encoders/ckpt0.pt'
img_encoder_path = 'models/autoencoders/Image/trained_img_encoders/ckpt0.pt'

# This will plot the EMG signal in the trial predicted force vs label force
PLOT_EMG = False
PLOT_HIST = True

# NN PARAMETERS
BATCH_SIZE = 8
EPOCHS = 5  # For Autoencoders
NN_EPOCHS = 5  # For End2End Models

CRITERION = 'MSE'  # choose from MSE/ L1/ HUBER
OPTIMIZER = 'Adam'  # choose from Adam/ SGD
LEARNING_RATE = 0.1
AUTO_LR = 0.0015  # Autoencoders Learning Rate
MOMENTUM = 0.9  # default = 0.9 / only for SGD

############################################################################################################
#           CHECK IF GPU IS AVAILABLE    #
############################################################################################################
#device = torch.device(device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
device = "cpu"
# device = torch.device(device=torch.device("cpu"))




############################################################################################################
#          ALL SETTINGS DONE           #
############################################################################################################


############################################################################################################
#          GENERATE DICTIONARIES          #
############################################################################################################
"""
If Code is changed in the future, I suggest to implement it such that new functions make use of these dictionaries.
USE PARAMETERS Dict which is initialized after the SAVING SETTING section 
"""

PREPROCESSING = {
    'resize': RESIZE,
    'img_size': IMG_SIZE,
    'hist_eq': HIST_EQ,
    'img_mean': IMG_MEAN,
    'img_std': IMG_STD,
    'emg_mean': EMG_MEAN,
    'emg_std': EMG_STD,
    'stripes': STRIPES,
    'stripe_width': STRIPE_WIDTH,
}

SETTINGS = {
    'mode': MODEL_TYPE,
    'train_val_split': TRAIN_VAL_SPLIT,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'criterion': CRITERION,
    'optimizer': OPTIMIZER,
    'lr': LEARNING_RATE,
    'momentum': MOMENTUM,
    'fake_trials': FAKE_TRIALS,
    'fake_percentage': FAKE_PERCENTAGE,
    'nn_epochs': NN_EPOCHS,
    'train' : TRAIN_AUTOENCODER,
    'validate': VALIDATE_AUTOENCODER,
    'plot_emg': PLOT_EMG,
    'debug': DEBUG,
    'img_encoder_path': img_encoder_path,
    'emg_encoder_path': emg_encoder_path,
    'device': device,
    'plot_hist': PLOT_HIST,

}

settings_configurator(MODEL_TYPE, SETTINGS)

############################################################################################################
###     Saving Settings   ###
############################################################################################################

IMG_PATH = 'models/autoencoders/Image/trained_img_encoders/'
EMG_PATH = 'models/autoencoders/EMG/trained_emg_encoders/'
MODEL_PATH = 'models/end2end/trained_e2e_models/'

IMG_CHECKPOINT = {
    'path': IMG_PATH,
    'prefix': 'ckpt',
    'counter': 0,
}
EMG_CHECKPOINT = {
    'path': EMG_PATH,
    'prefix': 'ckpt',
    'counter': 0,
}

MODEL_CHECKPOINT = {
    'path': MODEL_PATH,
    'prefix': 'ckpt',
    'counter': 0,
}

PARAMETERS = {'settings': SETTINGS, 'preprocessing': PREPROCESSING, 'img_checkpoint': IMG_CHECKPOINT,
              'emg_checkpoint': EMG_CHECKPOINT, 'model_checkpoint': MODEL_CHECKPOINT, 'time_stamp': TIME_STAMP}

############################################################################################################
#           GET PATHS           #
############################################################################################################

trial_paths = Directory.get_path().trial_path

############################################################################################################
#           GET TRIALS          #
############################################################################################################
trials = []
for path in trial_paths:
    trial = Trial.from_files(path, is_upper_case=True, preprocessing=PREPROCESSING)
    trials.append(trial)

TRAIN_TRIALS, VALIDATION_TRIALS = split_trials(trials, percentage=TRAIN_VAL_SPLIT)

if DEBUG:
    print(f' len trials at beginning: {len(trials)}')
    # TRAIN_TRIALS = trials[:-3]
    print(f'Length of train trials: {len(TRAIN_TRIALS)}')
    # VALIDATION_TRIALS = trials[-3:]
    print(f'Length of validation trials: {len(VALIDATION_TRIALS)}')

trials = [TRAIN_TRIALS, VALIDATION_TRIALS]

############################################################################################################
#           CREATE TRAIN UNITS    #
############################################################################################################

if FAKE_TRIALS:  # Check if model overfits
    trials = fake_forces(TRAIN_TRIALS, fake_percentage=FAKE_PERCENTAGE)
    train_units = func.map_emg_to_frame(trials)
    print('Fake trials created and mapped to frames')
    validation_units = func.map_emg_to_frame(VALIDATION_TRIALS)

else:
    train_units = func.map_emg_to_frame(TRAIN_TRIALS)
    validation_units = func.map_emg_to_frame(VALIDATION_TRIALS)

if STRIPES:
    train_units = func.get_image_stripes(train_units, nr_stripes=Nr_STRIPES, width=STRIPE_WIDTH, mode=STRIPES_MODE)
    validation_units = func.get_image_stripes(validation_units, nr_stripes=Nr_STRIPES, width=STRIPE_WIDTH,
                                              mode=STRIPES_MODE)

if EVEN_DATASET:
    train_units = even_dataset(train_units, max_force=MAX_FORCE, hist=PLOT_HIST)

############################################################################################################
#           SPLIT TRAIN UNITS    #
#############################################################################################################

LEN_TRAIN = len(train_units)
LEN_TEST = len(validation_units)

SETTINGS['train_len'] = LEN_TRAIN
SETTINGS['test_len'] = LEN_TEST
############################################################################################################
#           SPLIT TRAIN UNITS    #
############################################################################################################

train_ds = MyDataset(train_units)
val_ds = MyDataset(validation_units)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # only Train Trials
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)  # only Test Trials

############################################################################################################
#           INITIALIZE MODEL AND SET PARAMETERS    #
############################################################################################################
"""
In this section the model is initialized. Depending on the PARAMETERS['settings']['mode'] the respective model/models 
are initialized. All non end2end solution will use at least one Autoencoder and a regressor. 
"""

model, img_encoder, img_decoder, emg_encoder, emg_decoder = initialize_model(PARAMETERS)
regressor = setup_regressor(PARAMETERS)
criterion = choose_criterion(PARAMETERS)
initialize_optimizer(PARAMETERS)

############################################################################################################
#           TRAIN MODEL    #
############################################################################################################

MODEL_TYPE = PARAMETERS['settings']['mode']
if MODEL_TYPE == 'end2end':
    train_and_validate_end2end(train_loader, val_loader, TRAIN_TRIALS,VALIDATION_TRIALS, PARAMETERS)

elif MODEL_TYPE == 'encoded_US':
    train_and_validate_encoded_US(train_loader, val_loader, PARAMETERS, train_units, validation_units, trials)

elif MODEL_TYPE == 'emg_regression':
    train_and_validate_emg_regression(regressor, train_units, validation_units, trials, PARAMETERS)

elif MODEL_TYPE == 'encoded_regression':
    train_and_validate_encoded_regression(train_loader, val_loader, train_units, validation_units, trials, PARAMETERS)

elif MODEL_TYPE == 'encoded_end2end':
    train_and_validate_encoded_end2end(train_loader, val_loader, TRAIN_TRIALS, VALIDATION_TRIALS, PARAMETERS)
############################################################################################################
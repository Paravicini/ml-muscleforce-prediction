from sklearn.utils import shuffle
from models.regression.dataset_builder import get_dataset
from models.regression.regression_analysis import analyse_pred
import sys


def fit_regression(regressor, img_encoder, train_units, val_units, trials, parameters, emg_encoder=None ):
    print(f'Fitting {regressor.__class__.__name__}...')

    trainset = get_dataset(train_units, parameters, img_encoder, emg_encoder=emg_encoder,
                           ds_emg=parameters["settings"]['dataset_emg'], ds_frame=parameters["settings"]['dataset_frame'])

    trainset = shuffle(trainset)
    X = trainset.iloc[:, :-1]

    y = trainset.iloc[:, -1]

    # valset = get_dataset(val_units, params, img_encoder=img_encoder, emg_encoder=emg_encoder,
    # ds_emg=params["settings"]['dataset_emg'],
    # ds_frame=params["settings"]['dataset_frame'])
    # X_val = valset.iloc[:, :-1]
    # y_val = valset.iloc[:, -1]

    regressor.fit(X, y)

    # Measure the memory footprint
    memory_size = sys.getsizeof(regressor)
    print(f"XGB Regressor memory footprint: {memory_size} bytes")

    analyse_pred(regressor, trainset, trials, parameters, img_encoder=img_encoder, emg_encoder=emg_encoder)



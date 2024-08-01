from sklearn.utils import shuffle
from models.regression.dataset_builder import get_dataset
from models.regression.regression_analysis import analyse_pred


def train_and_validate_emg_regression(regressor, train_units, val_units, trials, params):
    print(f'Fitting {regressor.__class__.__name__}...')
    trainset = get_dataset(train_units, params, ds_emg=params["settings"]['dataset_emg'], ds_frame=params["settings"]['dataset_frame'])
    trainset = shuffle(trainset)
    X = trainset.iloc[:, :-1]
    y = trainset.iloc[:, -1]

    valset = get_dataset(val_units, params, ds_emg=params["settings"]['dataset_emg'], ds_frame=params["settings"]['dataset_frame'])
    valset = shuffle(valset)
    X_val = valset.iloc[:, :-1]
    y_val = valset.iloc[:, -1]

    regressor.fit(X, y)
    analyse_pred(regressor, trainset, trials, params)


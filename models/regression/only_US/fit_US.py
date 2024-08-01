from models.regression.dataset_builder import get_dataset
from models.regression.regression_analysis import analyse_pred


def fit_US_reg(regressor, img_encoder, train_units, trials, params, ):
    trainset = get_dataset(train_units, params, img_encoder= img_encoder, ds_emg=params["settings"]['dataset_emg'], ds_frame=params["settings"]['dataset_frame'])
    print(f'shape trainset: {trainset.shape}')
    X = trainset.iloc[:, :-1]
    y = trainset.iloc[:, -1]

    #valset = get_dataset(val_units, settings, img_encoder= img_encoder, ds_emg=settings['dataset_emg'], ds_frame=settings['dataset_frame'])
    #val_X = valset.iloc[:, :-1]
    #val_y = valset.iloc[:, -1]

    regressor.fit(X, y)

    analyse_pred(regressor, trainset, trials, params, img_encoder=img_encoder)

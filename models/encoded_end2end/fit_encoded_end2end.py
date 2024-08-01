import torch
from matplotlib import pyplot as plt
from models.helper_functions import save_model_bool, save_model


def fit_encoded_end2end(train_loader, val_loader, parameters):
    print(f'Encoded End2End training started')
    model = parameters["model"].train()
    checkpoint = parameters["model_checkpoint"]

    epoch_loss_rmse = []
    val_loss_rmse = []
    device = parameters["settings"]["device"]
    lowest_loss = 0.0
    for epoch in range(parameters["settings"]["nn_epochs"]):
        epoch_loss = 0.0
        epoch_squared_diffs = 0.0
        epoch_total_samples = 0.0
        for frame, emg, force_hat in train_loader:
            frame = frame.type(torch.float32).squeeze(1).to(device)
            emg = emg.type(torch.float32).squeeze(1).to(device)

            force_hat = force_hat.type(torch.float32).squeeze(1).to(device)
            parameters["optimizer"].zero_grad()
            outputs = model(emg, frame)
            loss = parameters["criterion"](outputs, force_hat)
            loss.backward()
            parameters["optimizer"].step()
            epoch_loss += loss.item()
            epoch_squared_diffs += torch.sum((outputs - force_hat) ** 2)
            epoch_total_samples += frame.shape[0]

        epoch_rmse = torch.sqrt(epoch_squared_diffs / epoch_total_samples).cpu().detach().numpy()
        epoch_loss_rmse.append(epoch_rmse)
        val_rmse = get_val_rmse(model, val_loader, parameters)
        val_loss_rmse.append(val_rmse)
        if save_model_bool(epoch, val_rmse, lowest_loss, parameters, threshold=(lowest_loss * 0.1)):
            lowest_loss = val_rmse
            best_epoch = epoch
            print(f'Saving model on epoch {epoch + 1}')
            checkpoint = save_model(model, parameters["optimizer"], epoch, checkpoint, loss)

        print(f"Epoch {epoch} - Training loss: {epoch_rmse} | Validation RMSE: {val_rmse}")
    plot_loss(epoch_loss_rmse, val_loss_rmse)
    print(f'Best epoch: {best_epoch:.2f} --> Val Score: {lowest_loss:.2f}')
    best_model_path = checkpoint['path'] + checkpoint['prefix'] + str(checkpoint['counter'] - 1) + '.pt'
    return best_model_path


def get_val_rmse(model, val_loader, params):
    squared_diffs = 0.0
    total_samples = 0.0
    device = params["settings"]["device"]
    for frame, emg, force_hat in val_loader:
        frame = frame.type(torch.float32).squeeze(1).to(device)
        emg = emg.type(torch.float32).squeeze(1).to(device)
        force_hat = force_hat.type(torch.float32).squeeze(1).to(device)

        outputs = model(emg, frame)
        squared_diffs += torch.sum((outputs - force_hat) ** 2)
        total_samples += frame.shape[0]
    rmse = torch.sqrt(squared_diffs / total_samples).cpu().detach().numpy()
    return rmse


def plot_loss(epoch_loss_rmse, val_loss_rmse):
    plt.plot(epoch_loss_rmse, label="Training RMSE")
    plt.plot(val_loss_rmse, label="Validation RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()




import torch


def get_validation_loss(model, val_loader, device):
    model.eval()
    val_running_loss = 0.0
    val_squared_diffs = 0.0
    val_total_samples = 0
    val_crit = torch.nn.MSELoss()
    with torch.no_grad():
        val_loss = 0
        for i, (frame, emg, force) in enumerate(val_loader):
            emg = emg.type(torch.float32).squeeze(1).to(device)
            frame = frame.type(torch.float32).squeeze(1).to(device)
            labels = force.type(torch.float32).squeeze(1).to(device)
            outputs = model(emg.float(), frame.float())
            loss = val_crit(outputs, labels)
            val_running_loss += loss.item()
            val_squared_diffs += torch.sum((outputs - labels) ** 2)
            val_total_samples += frame.shape[0]
    val_rmse = torch.sqrt(val_squared_diffs / val_total_samples).cpu().detach().numpy()
    return val_rmse


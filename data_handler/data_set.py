from torch.utils.data import Dataset, DataLoader


# Source: https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

class MyDataset(Dataset):
    """
    Custom Pytorch Dataset class, such that pytorch dataloader can be used.
    """
    def __init__(self, data):
        self.train_unit = data

    def __getitem__(self, index):
        frame = self.train_unit[index].frame
        emg = self.train_unit[index].emg
        force = self.train_unit[index].force
        return frame, emg, force

    def __len__(self):
        return len(self.train_unit)

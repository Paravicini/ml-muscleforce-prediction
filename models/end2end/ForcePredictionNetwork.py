"""
Structure from: https://ai.plainenglish.io/design-your-first-custom-neural-network-from-scratch-using-pytorch-a14ede6271ff
or: https://medium.com/plain-simple-software/the-pytorch-cnn-guide-for-beginners-c37808e2de03
or: https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
or: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
"""
import torch.nn as nn
from torch import cat


class ForcePredictionNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self):
        super(ForcePredictionNN, self).__init__()
        self.conv_emg = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=2, kernel_size=15, stride=2, padding=1),
                                      nn.LeakyReLU(),
                                      nn.BatchNorm1d(2),
                                      nn.MaxPool1d(kernel_size=2, stride=2),
                                      nn.Flatten())

        #for full frame
        #self.conv_frame_0 = nn.Sequential(
            #nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(10, 10), stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(1)
        #)

        self.conv_frame_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(15, 9), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(11, 5), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_frame_2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(7, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
        )

        self.flatt_frame = nn.Flatten()

        self.out = nn.Sequential(nn.Linear(84, 1),  # 5364,
                                 nn.Dropout(0.01), )

    # Progresses data across layers
    def forward(self, emg, frame):
        emg_out = self.conv_emg(emg)

        frame_out = self.conv_frame_1(frame)
        frame_out = self.conv_frame_2(frame_out)
        frame_out = self.flatt_frame(frame_out)


        out = (cat((emg_out, frame_out), dim=1))
        out = self.out(out)
        return out


